#based on https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wsc273/utils.py
#and https://huggingface.co/datasets/pakphum/winograd_th/discussions/2
def is_possessive(pronoun):
    # Check if the pronoun is a possessive form
    return pronoun.startswith("ของ")

def add_possessive(pronoun):
    return f"ของ{pronoun}"

def process_opt(option, pronoun):
    return add_possessive(option) if is_possessive(pronoun) else option


def process_docs(data):
    def process_fn(doc):
        pronoun = doc["pronoun"]
        #quote, ending = doc["text"][:doc["pronoun_loc"]], doc["text"][doc["pronoun_loc"]+len(pronoun):]
        options = [process_opt(opt, pronoun) for opt in doc["options"]]
        index = doc["pronoun_loc"] + len(doc["pronoun"])
        template = doc["text"][:doc["pronoun_loc"]]
        doc["query"] = doc["text"][index:]
        doc["choices"] = [f" {template}{option}" for option in options]
        return doc

    return data.map(process_fn)
