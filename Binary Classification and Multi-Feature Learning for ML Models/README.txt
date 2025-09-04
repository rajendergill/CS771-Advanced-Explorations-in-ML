### Prerequisites to run this code:
- Python 3.x
- pip or pip3
- Required Python packages: `tensorflow`, `numpy`, `pandas`, `sklearn`, `matplotlib`

Running the Code: Open VScode in this folder copy all the dataset(Training, Validation, Test datasets) files in this directory then run 51.py file to run the code this will generate all 4 prediction files(pred_emoticon.txt, pred_deepfeat.txt, pred_textseq.txt and
pred_combined.txt).
    
Instructions:
1.	Place all the files (51.py, model1.py, model2.py, model3.py, combine_model.py) in the same directory.
2.	Ensure you have the necessary data files(datasets files) are  available in the expected formats (e.g., .csv and .npz files) in the same directory.if they are not in the same directory the code will not run.

Note: 
If datasets are in different folders then give path of each dataset in our code in the beo lines of the code files below mentioned.
1. In model1.py file: 
    Line 83 :- 'train_emoticon.csv'  
    Line 84 :- 'valid_emoticon.csv'  
    Line 85 :- 'test_emoticon.csv'   

2. In model2.py file:
    Line 71 :- 'train_feature.npz'   
    Line 72 :- 'valid_feature.npz'   
    Line 73 :- 'test_feature.npz'    

3. In model3.py file:
    Line 75 :- 'train_text_seq.csv'
    Line 76 :- 'valid_text_seq.csv'
    Line 77 :- 'test_text_seq.csv'

4. In combine_model.py file:
    Line 12 :- 'train_emoticon.csv'
    Line 27 :- 'train_feature.npz'
    Line 31 :- 'train_text_seq.csv'
    Line 56 :- 'valid_emoticon.csv'
    Line 62 :- 'valid_feature.npz'
    Line 66 :- 'valid_text_seq.csv'
    Line 82 :- 'test_emoticon.csv'
    Line 88 :- 'test_feature.npz'
    Line 92 :- 'test_text_seq.csv'

In the above line you have to give the path of the datasets.

Note :- In case of any query regarding running the code please contact us at ( deepaksoni24@iitk.ac.in )
