import numpy as np
import cv2


# perspective transform on undistorted images
def perspective_transform(img):
    imshape = img.shape
    src = np.float32([[260, 680], [1080, 680], [600, 450], [690, 450]])
    dst = np.float32([[320, 680], [950, 680], [320, 0], [950, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    img_size = (imshape[1], imshape[0])
    perspective_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return perspective_img, Minv


def define_warper():

    src = np.float32([
        [255, 685],
        [1050, 685],
        [590, 455],
        [695, 455]
    ])

    dst = np.float32([
        [320, 685],
        [950, 685],
        [320, 0],
        [950, 0]
    ])

    return src, dst

# region of interest
def region_of_interest(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img, dtype=np.uint8)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image