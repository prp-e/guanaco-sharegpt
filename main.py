from datasets import load_dataset

def convert_to_sharegpt(example):
    text = example['text']
    if "### Human:" not in text or "### Assistant:" not in text:
        return {"conversations": []}
      
    parts = text.split("### Human:")

    
    content_parts = parts[1].split("### Assistant:")
    
    user_content = content_parts[0].strip()
    assistant_content = content_parts[1].strip()
    
    return {
        "conversations": [
            {
                "role" : "user",
                "content" : user_content
            },
            {
                "role" : "assistant",
                "content" : assistant_content
            }
        ]
    }

ds = load_dataset("timdettmers/openassistant-guanaco", split="train")
ds = ds.map(convert_to_sharegpt)