"""Email validation helpers."""

from __future__ import annotations

import contextlib
import re
import smtplib
import socket
import ssl
import time
import unicodedata
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import dns.exception
import dns.resolver

_NAME_TOKEN_RE = re.compile(r"[A-Za-z\u00C0-\u017F0-9']+")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")
_LOCAL_DISALLOWED_RE = re.compile(r"[^a-z0-9.\-_]")
_MULTI_SEP_RE = re.compile(r"[.\-_]{2,}")


class SMTPVerificationStatus(str, Enum):
    """High-level outcome of a passive SMTP verification."""

    VALIDE = "valide"
    DOUTEUX = "douteux"
    INVALIDE = "invalide"


@dataclass(frozen=True)
class SMTPVerificationResult:
    """Structured response for passive SMTP verification."""

    email: str
    domain: str
    status: SMTPVerificationStatus
    reason: Optional[str] = None
    mx: Optional[str] = None
    code: Optional[int] = None
    message: Optional[str] = None
    catch_all: Optional[bool] = None
    random_code: Optional[int] = None
    random_message: Optional[str] = None
    duration: Optional[float] = None


def _strip_accents(value: str) -> str:
    """Return *value* without diacritic marks."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _slugify_token(token: str) -> str:
    """Simplify a single name token to lowercase ASCII."""
    if not token:
        return ""
    ascii_token = _strip_accents(token)
    ascii_token = _NON_ALNUM_RE.sub("", ascii_token)
    return ascii_token.lower()


def _clean_local_part(value: str) -> str:
    """Keep email local-part safe characters and trim duplicated separators."""
    if not value:
        return ""
    clean = _LOCAL_DISALLOWED_RE.sub("", value.lower())
    clean = _MULTI_SEP_RE.sub(lambda match: match.group(0)[0], clean)
    clean = clean.lstrip("._-").rstrip("._-")
    return clean


def _normalize_domain(domain: str) -> str:
    """Extract and normalise a domain name from arbitrary input."""
    candidate = (domain or "").strip().lower()
    if not candidate:
        return ""
    candidate = candidate.replace("mailto:", "")
    candidate = candidate.split("@")[-1]
    candidate = candidate.strip()
    if not candidate:
        return ""
    url_like = candidate if "://" in candidate else f"http://{candidate}"
    parsed = urlparse(url_like)
    host = parsed.netloc or parsed.path
    host = host.strip()
    if host.startswith("www."):
        host = host[4:]
    host = host.split("/")[0]
    host = host.split(":")[0]
    host = host.strip().strip(".")
    return host


def _domain_has_mx(cleaned: str) -> bool:
    """Return True if *cleaned* exposes at least one MX record."""
    if not cleaned:
        return False
    try:
        answers = dns.resolver.resolve(cleaned, "MX")
        return len(answers) > 0
    except Exception:
        return False


def has_mx_record(domain: str) -> bool:
    """Return True if the domain exposes at least one MX record."""

    cleaned = _normalize_domain(domain)
    return _domain_has_mx(cleaned)


def generate_email_patterns(full_name: str, domain: str) -> List[str]:
    """
    Generate plausible corporate email addresses for ``full_name`` at ``domain``.

    The function emits a list of common patterns (``prenom.nom@``, ``p.nom@``,
    ``nom.prenom@``...) filtered to domains that expose an MX record.
    """

    domain_clean = _normalize_domain(domain)
    if not domain_clean or not _domain_has_mx(domain_clean):
        return []

    raw_tokens = [part for part in _NAME_TOKEN_RE.findall(full_name or "") if part.strip()]
    slug_tokens = [_slugify_token(token) for token in raw_tokens]
    slug_tokens = [token for token in slug_tokens if token]
    if not slug_tokens:
        return []

    first = slug_tokens[0]
    remaining = slug_tokens[1:] or [first]
    last = remaining[-1]
    last_tokens = remaining
    last_compound = "".join(last_tokens)
    last_dotted = ".".join(last_tokens) if len(last_tokens) > 1 else ""

    if not first or not last:
        return []

    first_initial = first[0]
    last_initial = last[0] if last else ""
    initials = "".join(token[0] for token in slug_tokens if token)

    candidates: List[str] = []
    seen: set[str] = set()

    def add(local_part: str) -> None:
        clean = _clean_local_part(local_part)
        if clean and clean not in seen:
            seen.add(clean)
            candidates.append(f"{clean}@{domain_clean}")

    add(f"{first}.{last}")
    add(f"{first}{last}")
    if last_dotted:
        add(f"{first}.{last_dotted}")
    if last_compound and last_compound != last:
        add(f"{first}{last_compound}")

    add(f"{first_initial}.{last}")
    add(f"{first_initial}{last}")
    if last_compound and last_compound != last:
        add(f"{first_initial}{last_compound}")

    add(f"{last}.{first}")
    add(f"{last}{first}")

    if last_initial:
        add(f"{first}.{last_initial}")
        add(f"{first}{last_initial}")
        add(f"{first_initial}.{last_initial}")
        add(f"{first_initial}{last_initial}")

    add(f"{first}-{last}")
    add(f"{first}_{last}")
    add(f"{last}-{first}")
    add(f"{last}_{first}")

    if last_compound and last_compound != last:
        add(last_compound)

    add(last)
    add(first)

    if initials and len(initials) > 1:
        add(initials)

    return candidates


def _decode_smtp_message(payload: object) -> str:
    """Convert SMTP byte responses to readable text."""
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return str(payload)


def _resolve_mx_records(domain: str, timeout: float) -> List[Tuple[int, str]]:
    """Resolve MX records for *domain* ordered by priority."""
    resolver = dns.resolver.Resolver()
    resolver.timeout = timeout
    resolver.lifetime = timeout
    answers = resolver.resolve(domain, "MX")
    records: List[Tuple[int, str]] = []
    for rdata in answers:
        preference = int(getattr(rdata, "preference", 0))
        exchange = str(getattr(rdata, "exchange", "")).rstrip(".")
        records.append((preference, exchange))
    records.sort(key=lambda item: item[0])
    return records


def _default_helo() -> str:
    """Return a safe default HELO hostname."""
    try:
        host = socket.getfqdn()
    except Exception:
        host = ""
    host = (host or "localhost").strip()
    return host or "localhost"


def _build_mail_from(domain: str) -> str:
    """Construct a neutral MAIL FROM envelope address."""
    clean_domain = domain.strip() or "localhost"
    return f"postmaster@{clean_domain}"


def _check_catch_all(
    smtp: smtplib.SMTP,
    mail_from: str,
    domain: str,
) -> Tuple[Optional[bool], Optional[int], Optional[str], Optional[str]]:
    """
    Probe catch-all behaviour by issuing a RCPT TO command with a random address.

    Returns a tuple (catch_all, code, message, error_reason).
    """

    try:
        smtp.rset()
        code_mail, _ = smtp.mail(mail_from)
        if code_mail >= 400:
            return None, None, None, "mail_from_rejected"
        bogus_local = f"nope-{uuid.uuid4().hex[:12]}"
        bogus_email = f"{bogus_local}@{domain}"
        code_random, message_random = smtp.rcpt(bogus_email)
        catch_all = 200 <= code_random < 300
        return catch_all, code_random, _decode_smtp_message(message_random), None
    except smtplib.SMTPResponseException as exc:
        return None, exc.smtp_code, _decode_smtp_message(exc.smtp_error), "smtp_response_error"
    except smtplib.SMTPException as exc:
        return None, None, str(exc), "smtp_exception"


def passive_smtp_verify(
    email: str,
    *,
    timeout: float = 10.0,
    helo_hostname: Optional[str] = None,
    mail_from: Optional[str] = None,
) -> SMTPVerificationResult:
    """
    Passively verify *email* via SMTP up to RCPT TO without sending a message.

    The function returns an :class:`SMTPVerificationResult` whose ``status`` is
    ``valide``, ``douteux`` or ``invalide``.  When the remote server accepts any
    address (catch-all), the ``catch_all`` flag is set to ``True`` and the status
    is ``douteux``.
    """

    trimmed = (email or "").strip()
    if "@" not in trimmed:
        return SMTPVerificationResult(
            email=trimmed,
            domain="",
            status=SMTPVerificationStatus.INVALIDE,
            reason="format",
        )

    local_part, domain = trimmed.rsplit("@", 1)
    local_part = local_part.strip()
    domain = domain.strip().lower()
    if not local_part or not domain:
        return SMTPVerificationResult(
            email=trimmed,
            domain=domain,
            status=SMTPVerificationStatus.INVALIDE,
            reason="format",
        )

    helo = helo_hostname or _default_helo()
    envelope_sender = mail_from or _build_mail_from(domain)

    try:
        mx_records = _resolve_mx_records(domain, timeout)
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
        return SMTPVerificationResult(
            email=trimmed,
            domain=domain,
            status=SMTPVerificationStatus.INVALIDE,
            reason="mx_not_found",
        )
    except dns.exception.DNSException as exc:
        return SMTPVerificationResult(
            email=trimmed,
            domain=domain,
            status=SMTPVerificationStatus.DOUTEUX,
            reason="mx_lookup_error",
            message=str(exc),
        )

    if not mx_records:
        return SMTPVerificationResult(
            email=trimmed,
            domain=domain,
            status=SMTPVerificationStatus.INVALIDE,
            reason="mx_not_found",
        )

    last_error: Optional[SMTPVerificationResult] = None
    for _, mx_host in mx_records:
        start = time.time()
        smtp: Optional[smtplib.SMTP] = None
        try:
            smtp = smtplib.SMTP(mx_host, 25, local_hostname=helo, timeout=timeout)
            smtp.set_debuglevel(0)
            smtp.ehlo_or_helo_if_needed()
            if smtp.has_extn("starttls"):
                context = ssl.create_default_context()
                smtp.starttls(context=context)
                smtp.ehlo_or_helo_if_needed()

            code_mail, message_mail = smtp.mail(envelope_sender)
            if code_mail >= 500:
                return SMTPVerificationResult(
                    email=trimmed,
                    domain=domain,
                    status=SMTPVerificationStatus.DOUTEUX,
                    reason="mail_from_rejected",
                    mx=mx_host,
                    code=code_mail,
                    message=_decode_smtp_message(message_mail),
                    duration=round(time.time() - start, 4),
                )

            code_rcpt, message_rcpt = smtp.rcpt(trimmed)
            decoded_rcpt = _decode_smtp_message(message_rcpt)

            if 500 <= code_rcpt:
                return SMTPVerificationResult(
                    email=trimmed,
                    domain=domain,
                    status=SMTPVerificationStatus.INVALIDE,
                    reason="rcpt_denied",
                    mx=mx_host,
                    code=code_rcpt,
                    message=decoded_rcpt,
                    duration=round(time.time() - start, 4),
                )

            if 400 <= code_rcpt < 500:
                return SMTPVerificationResult(
                    email=trimmed,
                    domain=domain,
                    status=SMTPVerificationStatus.DOUTEUX,
                    reason="rcpt_tempfail",
                    mx=mx_host,
                    code=code_rcpt,
                    message=decoded_rcpt,
                    duration=round(time.time() - start, 4),
                )

            if code_rcpt == 252:
                return SMTPVerificationResult(
                    email=trimmed,
                    domain=domain,
                    status=SMTPVerificationStatus.DOUTEUX,
                    reason="rcpt_unverified",
                    mx=mx_host,
                    code=code_rcpt,
                    message=decoded_rcpt,
                    duration=round(time.time() - start, 4),
                )

            if 200 <= code_rcpt < 300:
                catch_all, random_code, random_message, error_reason = _check_catch_all(
                    smtp,
                    envelope_sender,
                    domain,
                )
                if catch_all is True:
                    return SMTPVerificationResult(
                        email=trimmed,
                        domain=domain,
                        status=SMTPVerificationStatus.DOUTEUX,
                        reason="catch_all",
                        mx=mx_host,
                        code=code_rcpt,
                        message=decoded_rcpt,
                        catch_all=True,
                        random_code=random_code,
                        random_message=random_message,
                        duration=round(time.time() - start, 4),
                    )

                if catch_all is False:
                    return SMTPVerificationResult(
                        email=trimmed,
                        domain=domain,
                        status=SMTPVerificationStatus.VALIDE,
                        reason="accepted",
                        mx=mx_host,
                        code=code_rcpt,
                        message=decoded_rcpt,
                        catch_all=False,
                        random_code=random_code,
                        random_message=random_message,
                        duration=round(time.time() - start, 4),
                    )

                # Inconclusive catch-all probe falls back to doubtful result.
                return SMTPVerificationResult(
                    email=trimmed,
                    domain=domain,
                    status=SMTPVerificationStatus.DOUTEUX,
                    reason=error_reason or "catch_all_inconclusive",
                    mx=mx_host,
                    code=code_rcpt,
                    message=decoded_rcpt,
                    catch_all=None,
                    random_code=random_code,
                    random_message=random_message,
                    duration=round(time.time() - start, 4),
                )

            return SMTPVerificationResult(
                email=trimmed,
                domain=domain,
                status=SMTPVerificationStatus.DOUTEUX,
                reason="rcpt_unexpected_code",
                mx=mx_host,
                code=code_rcpt,
                message=decoded_rcpt,
                duration=round(time.time() - start, 4),
            )
        except (smtplib.SMTPConnectError, smtplib.SMTPHeloError) as exc:
            last_error = SMTPVerificationResult(
                email=trimmed,
                domain=domain,
                status=SMTPVerificationStatus.DOUTEUX,
                reason="connection_error",
                mx=mx_host,
                message=str(exc),
            )
        except (socket.timeout, TimeoutError):
            last_error = SMTPVerificationResult(
                email=trimmed,
                domain=domain,
                status=SMTPVerificationStatus.DOUTEUX,
                reason="timeout",
                mx=mx_host,
            )
        except smtplib.SMTPResponseException as exc:
            last_error = SMTPVerificationResult(
                email=trimmed,
                domain=domain,
                status=SMTPVerificationStatus.DOUTEUX,
                reason="smtp_error",
                mx=mx_host,
                code=exc.smtp_code,
                message=_decode_smtp_message(exc.smtp_error),
            )
        except smtplib.SMTPException as exc:
            last_error = SMTPVerificationResult(
                email=trimmed,
                domain=domain,
                status=SMTPVerificationStatus.DOUTEUX,
                reason="smtp_exception",
                mx=mx_host,
                message=str(exc),
            )
        except OSError as exc:
            last_error = SMTPVerificationResult(
                email=trimmed,
                domain=domain,
                status=SMTPVerificationStatus.DOUTEUX,
                reason="os_error",
                mx=mx_host,
                message=str(exc),
            )
        finally:
            if smtp is not None:
                with contextlib.suppress(Exception):
                    smtp.quit()

    if last_error is not None:
        return last_error

    return SMTPVerificationResult(
        email=trimmed,
        domain=domain,
        status=SMTPVerificationStatus.DOUTEUX,
        reason="mx_hosts_unreachable",
    )


__all__ = [
    "SMTPVerificationResult",
    "SMTPVerificationStatus",
    "generate_email_patterns",
    "has_mx_record",
    "passive_smtp_verify",
]
