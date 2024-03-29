import cv2
import os


classNames2 = ["10 of clubs", "10 of diamonds", "10 of hearts", "10 of spades", "2 of clubs", "2 of diamonds",
               "2 of hearts", "2 of spades", "3 of clubs", "3 of diamonds", "3 of hearts", "3 of spades", "4 of clubs",
                 "4 of diamonds", "4 of hearts", "4 of spades", "5 of clubs", "5 of diamonds", "5 of hearts", "5 of spades",
                   "6 of clubs", "6 of diamonds", "6 of hearts", "6 of spades", "7 of clubs", "7 of diamonds", "7 of hearts",
                     "7 of spades", "7 of spades", "8 of clubs", "8 of diamonds", "8 of hearts", "8 of spades", "9 of clubs",
                       "9 of diamonds", "9 of hearts", "9 of spades", "Ace of clubs", "Ace of diamonds", "Ace of hearts", "Ace of spades",
                         "Jack of clubs", "Jack of diamonds", "Jack of hearts", "Jack of spades", "Joker", "King of clubs", "King of diamonds",
                           "King of hearts", "King of spades", "Queen of clubs", "Queen of diamonds", "Queen of hearts", "Queen of spades",
                             "black chip", "blue chip", "card back", "chips", "green chip", "red chip", "white chip"]


file_name = os.path.join(os.path.dirname(__file__), '/test/images/6_jpg.rf.0f0fe570e17badaab037f36ec3be1c0d.jpg')
image = cv2.imread(file_name)

print(image.shape)