from openai import OpenAI
client = OpenAI()

vector_store_files = client.beta.vector_stores.files.list(
  vector_store_id="vs_abc123"
)
print(vector_store_files)
