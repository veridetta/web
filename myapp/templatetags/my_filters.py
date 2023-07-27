from django import template

register = template.Library()

@register.filter
def replace(value, args):
    search_string, replacement_string = args.split(',')
    return value.replace(search_string, replacement_string)
