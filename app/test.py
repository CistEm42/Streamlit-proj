from groq import Groq

client = Groq(api_key="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")

models = client.models.list()
for m in models.data:
    print(m.id)