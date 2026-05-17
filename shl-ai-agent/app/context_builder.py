def build_conversation_context(messages):

    context = []

    for msg in messages:
        role = msg.role.upper()
        context.append(f"{role}: {msg.content}")

    return "\n".join(context)