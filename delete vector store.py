from openai import OpenAI
client = OpenAI()

deleted_vector_store_file = client.beta.vector_stores.files.delete(
    vector_store_id="vs_abc123",
    file_id="file-abc123"
)
print(deleted_vector_store_file)
