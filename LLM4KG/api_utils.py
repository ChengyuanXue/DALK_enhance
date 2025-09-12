import time
from openai import OpenAI

# 配置第三方API客户端
client = OpenAI(
    api_key="sk-P4hNAfoKF4JLckjCuE99XbaN4bZIORZDPllgpwh6PnYWv4cj",
    base_url="https://aiyjg.lol/v1"
)

def request_api_palm(messages):
    """
    原来调用PaLM，现在改为调用第三方OpenAI兼容接口
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant to answer questions about biomedicine."},
                {"role": "user", "content": messages}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        ret = completion.choices[0].message.content.strip()
        return ret
    except Exception as E:
        print(f"API调用失败: {E}")
        time.sleep(2)
        return request_api_palm(messages)

def request_api_chatgpt(prompt):
    """
    ChatGPT API调用函数
    """
    messages = [
        {"role": "system", "content": 'You are an AI assistant to answer question about biomedicine.'},
        {"role": "user", "content": prompt}
    ]
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        ret = completion.choices[0].message.content.strip()
        return ret
    except Exception as E:
        time.sleep(2)
        print(f"API调用失败: {E}")
        return request_api_chatgpt(prompt)