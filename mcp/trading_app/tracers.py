import logging
import secrets
import string

from agents import Span, Trace, TracingProcessor

from database import write_log

logger = logging.getLogger(__name__)

ALPHANUM = string.ascii_lowercase + string.digits
TRACE_ID_BODY_LENGTH = 32
SEPARATOR = "0"


def make_trace_id(tag: str) -> str:
    """
    Return a string of the form 'trace_<tag>0<random>',
    where the total length after 'trace_' is 32 chars.
    """
    tag = tag.lower()
    if not tag.isalpha() or len(tag) >= TRACE_ID_BODY_LENGTH:
        raise ValueError(f"Trace tag must be letters only and shorter than {TRACE_ID_BODY_LENGTH} chars: {tag!r}")
    tag += SEPARATOR
    random_suffix = "".join(secrets.choice(ALPHANUM) for _ in range(TRACE_ID_BODY_LENGTH - len(tag)))
    return f"trace_{tag}{random_suffix}"


class LogTracer(TracingProcessor):
    def __init__(self, known_names: list[str]):
        # Only traces created by make_trace_id() for a known trader are logged; random trace ids
        # (hex, which usually contain '0') would otherwise produce garbage log names.
        self.known_names = {name.lower() for name in known_names}

    def get_name(self, trace_or_span: Trace | Span) -> str | None:
        body = trace_or_span.trace_id.removeprefix("trace_")
        name, separator, _ = body.partition(SEPARATOR)
        return name if separator and name in self.known_names else None

    def _write(self, name: str, log_type: str, message: str) -> None:
        # Tracing must never break an agent run.
        try:
            write_log(name, log_type, message)
        except Exception:
            logger.exception("Failed to write trace log")

    def _span_event(self, span, event: str) -> None:
        name = self.get_name(span)
        if not name:
            return
        data = span.span_data
        log_type = data.type if data and data.type else "span"
        parts = [event]
        if data:
            parts += [str(value) for value in (data.type, getattr(data, "name", None), getattr(data, "server", None)) if value]
        if span.error:
            parts.append(str(span.error))
        self._write(name, log_type, " ".join(parts))

    def on_trace_start(self, trace) -> None:
        if name := self.get_name(trace):
            self._write(name, "trace", f"Started: {trace.name}")

    def on_trace_end(self, trace) -> None:
        if name := self.get_name(trace):
            self._write(name, "trace", f"Ended: {trace.name}")

    def on_span_start(self, span) -> None:
        self._span_event(span, "Started")

    def on_span_end(self, span) -> None:
        self._span_event(span, "Ended")

    def force_flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass
