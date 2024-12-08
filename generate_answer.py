from ctransformers import AutoModelForCausalLM
import pandas as pd

# Path to your GGUF model
model_path = "D:/LLMs/vietnamese-llama2-7b-40gb.Q4_K_M.gguf"

# Load your CSV file
retrieval_csv_save = pd.read_csv("D:/GitHub Repos/UET_Chatbot_QA/better_retrieval_result_1-800.csv")

# Load the model
gen_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    model_type="llama"  # Specify the model type
)

# Prepare the prompt template
prompt_template = (
    "### System:\n"
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

def get_prompt(question, contexts):
    """
    Creates the complete prompt for the model.
    """
    context = "\n\n".join([f"Context [{i+1}]: {text}" for i, text in enumerate(contexts)])
    instruction = ('Bạn là trợ lý AI giúp sinh viên nộp đơn vào Đại học Kỹ thuật và Công nghệ Việt Nam. '
                   'Cung cấp câu trả lời chi tiết để người nộp đơn yêu cầu thông tin không cần phải tìm kiếm bên ngoài.')
    input_text = (
        f"Dựa vào một số đoạn văn về quy chế đào tạo của trường Đại học Công Nghệ Đại học Quốc Gia Hà Nội được cho dưới đây, "
        f"trả lời câu hỏi ở cuối.\n\n{context}\n\nQuestion: {question}\nHãy trả lời chi tiết và đầy đủ."
    )
    return prompt_template.format(instruction=instruction, input=input_text)

# Generate a response
def generate_response(prompt, max_new_tokens=4000):
    """
    Generates a response for the given prompt using the model.
    """
    return gen_model(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9
    )

# Extract contexts from the 'text' column
question = "Hệ thống tổ chức và quản lý đào tạo đại học tại ĐHQGHN gồm mấy cấp?"
contexts = retrieval_csv_save['text'].head(5).tolist()  # Get the first 5 contexts from the 'text' column

# Generate the prompt
prompt = get_prompt(question, contexts)

# Generate the response
response = generate_response(prompt)
print(response)
