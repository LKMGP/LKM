try:
    import termcolor
    has_termcolor = True
except:
    has_termcolor = False

try:
    import config
    color_scheme = config.COLOR_SCHEME
except:
    color_scheme = 'dark'

def paren_colors():
    if color_scheme == 'dark':
        return ['red', 'green', 'cyan', 'magenta', 'yellow']
    elif color_scheme == 'light':
        return ['blue', 'red', 'magenta', 'green', 'cyan']
    else:
        raise RuntimeError('Unknown color scheme: %s' % color_scheme)

def colored(text, depth):
    if has_termcolor:
        colors = paren_colors()
        color = colors[depth % len(colors)]
        return termcolor.colored(text, color, attrs=['bold'])
    else:
        return text

def format_if_possible(format, value):
    try:
        return format % value
    except:
        return '%s' % value
