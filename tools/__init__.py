from .search import search

def call_tools(tool_name, tool_input):
    if tool_name == "search":
        return search(tool_input)
    else:
        raise ValueError("Unknown tool name: {}".format(tool_name))
