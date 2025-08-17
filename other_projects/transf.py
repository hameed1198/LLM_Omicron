from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("Thoday weather is not good!"))
