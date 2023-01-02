from PIL import Image

im = Image.open("./img1.png")
# Lets display the image inline
display(im)
# Lets now convert it to black and white and segment it up into lines with kraken
bounding_boxes = pageseg.segment(im.convert("1"))["boxes"]
# And lets print those lines to the screen
print(bounding_boxes)
