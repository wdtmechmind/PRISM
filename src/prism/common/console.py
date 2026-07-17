try:
    from rich.console import Console
    from rich.prompt import Confirm
    from rich.text import Text
except ImportError:  # pragma: no cover - exercised only when rich is absent on a capture host.
    Console = None
    Confirm = None
    Text = None


_CONSOLE = Console(emoji=True) if Console is not None else None


def _plain(message, icon=''):
    prefix = (icon + ' ') if icon else ''
    print(prefix + message)


def emit(message, icon='', style=''):
    if _CONSOLE is None:
        _plain(message, icon)
        return
    if icon:
        _CONSOLE.print(icon, Text(' ' + message, style=style))
    else:
        _CONSOLE.print(Text(message, style=style))


def rule(title):
    if _CONSOLE is None:
        _plain('--- %s ---' % title)
        return
    _CONSOLE.rule(Text(title, style='bold cyan'))


def info(message):
    emit(message, '📡', 'cyan')


def step(message):
    emit(message, '▶️', 'bold blue')


def success(message):
    emit(message, '✅', 'green')


def warning(message):
    emit(message, '⚠️', 'yellow')


def saved(message):
    emit(message, '💾', 'green')


def done(message):
    emit(message, '🎉', 'bold green')


def ask_yes_no(message, default=False):
    if Confirm is None or _CONSOLE is None:
        suffix = ' [Y/n]: ' if default else ' [y/N]: '
        answer = input(message + suffix)
        if not answer.strip():
            return default
        return answer.strip().lower() in ['y', 'yes', '1', 'true']
    return Confirm.ask('🧪 ' + message, default=default, console=_CONSOLE)