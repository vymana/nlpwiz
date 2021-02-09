
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"


# Load Model
# pretrained = 'facebook/bart-large-xsum'
pretrained = "sshleifer/distilbart-xsum-12-3"
model = BartForConditionalGeneration.from_pretrained(pretrained)
tokenizer = BartTokenizer.from_pretrained(pretrained)

model.to(device)
model.eval()


def summarize(text, max_len, num_beams, ):
    if not text:
        return text

    inputs = tokenizer.batch_encode_plus(
        [text], max_length=1024, return_tensors="pt"
    )
    inputs = inputs.to(device)

    # Generate Summary
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=num_beams,
        max_length=max_len,
        early_stopping=True,
    )
    out = [
        tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for g in summary_ids
    ]

    return out

if __name__ == "__main__":
    text = """
    Peter and Elizabeth took a taxi to attend the night party in the city. While in the party, Elizabeth collapsed and was rushed to the hospital. Since she was diagnosed with a brain injury, the doctor told Peter to stay besides her until she gets well. Therefore, Peter stayed with her at the hospital for 3 days without leaving.
    """
    summarize(text, 50, 4)
