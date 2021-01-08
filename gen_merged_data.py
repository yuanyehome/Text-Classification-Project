import pandas as pd

for split in ["train", "test", "dev"]:
    res = {
        "label": [],
        "text": [],
        "title": [],
        "description": []
    }
    data = pd.read_csv("./data/%s.csv" % split)
    labels = data["Class Index"]
    titles = data["Title"]
    descriptions = data["Description"]
    for label, title, description in zip(labels, titles, descriptions):
        title = title.replace('\\', ' ')
        description = description\
            .replace('\\', ' ')\
            .replace('#39;', '\'')\
            .replace('#36;', '$')
        res["label"].append(label)
        res["title"].append(title)
        res["description"].append(description)
        res["text"].append("%s %s" % (title, description))
    res = pd.DataFrame(res)
    res.to_csv("merged_data/%s.csv" % split, index=False)
