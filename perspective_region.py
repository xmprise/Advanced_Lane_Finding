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


# region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img, dtype=np.uint8)
    # 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image