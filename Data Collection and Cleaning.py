data = pd.read_csv("Enter the file.txt", delimiter="\t", names=["english", "arabic"])
data.dropna(inplace=True)
