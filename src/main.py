import textsimilarity
import train

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #finetune model on new dataset
    train.train_model("./sentence similarity dataset.csv")   #




    #function to compare two files
    # df = text_similarity.get_similarity(doc1_path,doc2_path)


    #funtion to compare file against whole directory
    doc1_path = "/home/ecruz/projects/Plagiotron/g0pB_taske.txt"
    doc2_path = "/home/ecruz/projects/Plagiotron/g0pE_taske.txt"
    output_path = "./data/"

    df = textsimilarity.get_similarity_dir('./data/','./data/g0pE_taske.txt')
    df.to_csv(output_path+"output.csv")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
