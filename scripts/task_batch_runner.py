#!/usr/bin/env python3
"""Execute des taches decrites dans un fichier YAML par lots paralleles."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import yaml


LOGGER = logging.getLogger("task_batch_runner")


class TaskExecutionError(Exception):
    """Erreur remontee lorsqu'une tache echoue."""


class BatchAbort(Exception):
    """Exception interne utilisee pour interrompre les executions restantes."""


@dataclass
class TaskResult:
    block: str
    name: str
    task_type: str
    success: bool
    duration: float
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute des lots de taches decrits dans un fichier YAML."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Chemin vers le fichier YAML decrivant les blocs de taches.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Nombre de taches a executer en parallele (prioritaire sur le YAML).",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Arrete le traitement des la premiere tache en echec.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Niveau de log (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier YAML introuvable: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("La configuration YAML doit contenir un objet de niveau superieur.")
    return data


def chunked(sequence: Sequence[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for index in range(0, len(sequence), size):
        yield list(sequence[index : index + size])


def execute_block(
    block_name: str,
    tasks: Sequence[Dict[str, Any]],
    batch_size: int,
    stop_on_failure: bool,
) -> List[TaskResult]:
    LOGGER.info("Bloc '%s' (%s taches) - batch_size=%s", block_name, len(tasks), batch_size)
    results: List[TaskResult] = []

    for batch_index, batch_tasks in enumerate(chunked(list(tasks), batch_size), start=1):
        LOGGER.info("Bloc '%s' - lancement du batch %s (%s taches)", block_name, batch_index, len(batch_tasks))

        with ThreadPoolExecutor(max_workers=len(batch_tasks)) as executor:
            future_map: Dict[Future[TaskResult], Dict[str, Any]] = {}
            for task in batch_tasks:
                future_map[executor.submit(execute_task, block_name, task)] = task

            for future in as_completed(future_map):
                try:
                    result = future.result()
                    results.append(result)
                    if not result.success and stop_on_failure:
                        raise BatchAbort(f"Tache en echec: {result.block}/{result.name}")
                except BatchAbort:
                    raise
                except Exception as exc:  # pragma: no cover - garde-fou
                    task_info = future_map[future]
                    task_name = task_info.get("name") or task_info.get("command") or task_info.get("callable") or "task"
                    LOGGER.exception("Bloc '%s' - erreur inattendue dans la tache '%s'", block_name, task_name)
                    results.append(
                        TaskResult(
                            block=block_name,
                            name=str(task_name),
                            task_type=str(task_info.get("type", "inconnu")),
                            success=False,
                            duration=0.0,
                            error=str(exc),
                        )
                    )
                    if stop_on_failure:
                        raise BatchAbort(f"Tache en echec: {task_name}") from exc

    return results


def execute_task(block_name: str, task: Mapping[str, Any]) -> TaskResult:
    task_type = task.get("type")
    task_name = extract_task_name(task)
    prefix = f"[{block_name}/{task_name}]"
    start = time.perf_counter()

    LOGGER.info("%s Debut (%s)", prefix, task_type)

    try:
        if task_type == "shell":
            run_shell_task(prefix, task)
        elif task_type == "python":
            run_python_task(prefix, task)
        else:
            raise TaskExecutionError(f"Type de tache inconnu: {task_type!r}")
    except Exception as exc:
        duration = time.perf_counter() - start
        LOGGER.error("%s Echec apres %.2fs: %s", prefix, duration, exc)
        return TaskResult(
            block=block_name,
            name=task_name,
            task_type=str(task_type),
            success=False,
            duration=duration,
            error=str(exc),
        )

    duration = time.perf_counter() - start
    LOGGER.info("%s Termine en %.2fs", prefix, duration)
    return TaskResult(
        block=block_name,
        name=task_name,
        task_type=str(task_type),
        success=True,
        duration=duration,
    )


def extract_task_name(task: Mapping[str, Any]) -> str:
    if "name" in task and task["name"]:
        return str(task["name"])
    for key in ("command", "callable"):
        if key in task and task[key]:
            return str(task[key])
    return "tache"


def run_shell_task(prefix: str, task: Mapping[str, Any]) -> None:
    command = task.get("command")
    if not command:
        raise TaskExecutionError("Les taches de type 'shell' doivent definir 'command'.")

    cwd = task.get("cwd")
    env = task.get("env")
    shell_flag = False
    popen_args: Dict[str, Any] = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True, "bufsize": 1}

    if isinstance(command, str):
        shell_flag = True
        popen_args["args"] = command
    elif isinstance(command, Sequence):
        popen_args["args"] = list(command)
    else:
        raise TaskExecutionError("Le champ 'command' doit etre une chaine ou une liste.")

    if cwd:
        popen_args["cwd"] = str(cwd)
    if env:
        if not isinstance(env, Mapping):
            raise TaskExecutionError("'env' doit etre un dictionnaire cle/valeur.")
        merged_env = {**os.environ, **{str(key): str(value) for key, value in dict(env).items()}}
        popen_args["env"] = merged_env

    LOGGER.debug("%s Commande: %s", prefix, command)

    process = subprocess.Popen(shell=shell_flag, **popen_args)

    def stream(pipe, level):
        for line in iter(pipe.readline, ""):
            LOGGER.log(level, "%s %s", prefix, line.rstrip())
        pipe.close()

    threads = [
        threading.Thread(target=stream, args=(process.stdout, logging.INFO), daemon=True),
        threading.Thread(target=stream, args=(process.stderr, logging.ERROR), daemon=True),
    ]
    for thread in threads:
        thread.start()
    return_code = process.wait()
    for thread in threads:
        thread.join()

    if return_code != 0:
        raise TaskExecutionError(f"Commande shell terminee avec le code {return_code}.")


def run_python_task(prefix: str, task: Mapping[str, Any]) -> None:
    target = task.get("callable")
    if not target:
        raise TaskExecutionError("Les taches de type 'python' doivent definir 'callable'.")

    module_name, separator, func_path = str(target).partition(":")
    if not separator:
        raise TaskExecutionError("Le champ 'callable' doit suivre le format 'module:fonction'.")

    module = importlib.import_module(module_name)
    callable_obj: Any = module
    for attribute in func_path.split("."):
        if not attribute:
            raise TaskExecutionError(f"Chemin de fonction invalide dans 'callable': {target!r}")
        callable_obj = getattr(callable_obj, attribute)

    if not callable(callable_obj):
        raise TaskExecutionError(f"La cible {target!r} n'est pas appelable.")

    args = task.get("args", [])
    kwargs = task.get("kwargs", {})

    if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
        raise TaskExecutionError("'args' doit etre une liste ou un tuple.")
    if not isinstance(kwargs, Mapping):
        raise TaskExecutionError("'kwargs' doit etre un dictionnaire.")

    LOGGER.debug("%s Appel Python: %s args=%s kwargs=%s", prefix, target, args, kwargs)

    result = callable_obj(*args, **kwargs)
    if result is not None:
        serialized = repr(result)
        truncated = serialized if len(serialized) < 300 else serialized[:297] + "..."
        LOGGER.info("%s Resultat: %s", prefix, truncated)


def summarise_results(results: Sequence[TaskResult]) -> None:
    success_count = sum(result.success for result in results)
    failure_count = len(results) - success_count
    LOGGER.info("Taches reussies: %s | Taches en echec: %s", success_count, failure_count)
    if failure_count:
        for failed in results:
            if failed.success:
                continue
            LOGGER.error(
                "Echec: bloc='%s' tache='%s' type='%s' erreur='%s'",
                failed.block,
                failed.name,
                failed.task_type,
                failed.error,
            )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        config = load_yaml_config(args.config)
    except Exception as exc:
        LOGGER.error("Impossible de charger la configuration: %s", exc)
        return 1

    default_batch_size = config.get("batch_size", 1)
    try:
        batch_size = int(args.batch_size or default_batch_size)
    except (TypeError, ValueError):
        LOGGER.error("Valeur de batch_size invalide.")
        return 1
    if batch_size < 1:
        LOGGER.error("batch_size doit etre superieur ou egal a 1.")
        return 1

    blocks = config.get("blocks")
    if not blocks:
        LOGGER.error("La configuration doit definir une cle 'blocks' contenant la liste des blocs.")
        return 1

    all_results: List[TaskResult] = []

    try:
        for index, block in enumerate(blocks, start=1):
            block_name = str(block.get("name") or f"bloc-{index}")
            block_tasks = block.get("tasks") or []
            if not isinstance(block_tasks, Sequence):
                LOGGER.error("Le bloc '%s' doit contenir une liste 'tasks'.", block_name)
                return 1
            block_batch_size = int(block.get("batch_size", batch_size))
            block_results = execute_block(block_name, list(block_tasks), block_batch_size, args.stop_on_failure)
            all_results.extend(block_results)
    except BatchAbort as abort_exc:
        LOGGER.error("Arret anticipe: %s", abort_exc)
    except Exception as exc:  # pragma: no cover - securite supplementaire
        LOGGER.exception("Erreur inattendue lors de l'execution: %s", exc)
        return 1

    summarise_results(all_results)

    if any(not result.success for result in all_results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
