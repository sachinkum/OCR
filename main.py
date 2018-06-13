from userimageski import UserData
from data_fetch import OcrData
if __name__ == '__main__':
    # CREATES AN INSTANCE OF THE CLASS LOADING THE OCR DATA
    # Comment the following lines of code after model generation.
    data = OcrData('/media/sachin/Data/Workspace/Python/OCR/ocr-config.py')
    #
    # PERFORMS GRID SEARCH CROSS VALIDATION GETTING BEST MODEL OUT OF PASSED PARAMETERS
    data.perform_grid_search_cv('linearsvc-hog')
    #
    # TAKES THE PARAMETERS LINKED TO BEST MODEL AND RE-TRAINS THE MODEL ON THE WHOLE TRAIN SET
    data.generate_best_hog_model()
    #
    # TAKES THE JUST GENERATED MODEL AND EVALUATES IT ON TRAIN SET
    data.evaluate('/media/sachin/Data/Workspace/Python/OCR/linearsvc-hog-fulltrain36-90.pickle')

    # The following code performs the basic operation
    # creates instance of class and loads image
    n = int(input("Number of images?"))
    for i in range(1, n+1):
        user = UserData(str(i)+'.jpg')
        # detects objects in preprocessed image
        candidates = user.get_text_candidates()
        # selects objects containing text
        maybe_text = user.select_text_among_candidates(
            '/media/sachin/Data/Workspace/Python/OCR/linearsvc-hog-fulltrain2-90.pickle')
        # classifies single characters
        classified = user.classify_text('/media/sachin/Data/Workspace/Python/OCR/linearsvc-hog-fulltrain36-90.pickle')
        user.realign_text()

