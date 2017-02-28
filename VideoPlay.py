import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from preprocessing_module import *
from calibration_module import *
from sliding_window import *
import matplotlib.image as mpimg


def draw_on_original(undist, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Low accurate line
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))

    # Estimation is accurate line
    pts_left = np.array([np.transpose(np.vstack([left_fitx[400:], ploty[400:]]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx[400:], ploty[400:]])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.4, 0)
    return result


def processing(img):

    undist = get_camera_calibration(img)

    thresh_combined, grad_th, col_th = thresholding(undist)

    perspective, Minv = perspective_transform(thresh_combined)
    perspective = cv2.cvtColor(perspective, cv2.COLOR_RGB2GRAY).astype(np.uint8)

    slides_pers, left_fitx, right_fitx, ploty, avg_cur, dist_centre_val = sliding_window(perspective)

    mapped_lane = draw_on_original(undist, left_fitx, right_fitx, ploty, Minv)
    # font and text for drawing the offset and curvature
    curvature = "Estimated lane curvature %.2fm" % (avg_cur)
    dist_centre = "Estimated offset from lane center %.2fm" % (dist_centre_val)
    font = cv2.FONT_HERSHEY_COMPLEX

    cv2.putText(mapped_lane, curvature, (30, 60), font, 1.2, (255, 0, 0), 2)
    cv2.putText(mapped_lane, dist_centre, (30, 120), font, 1.2, (255, 0, 0), 2)

    return mapped_lane

if __name__ == '__main__':

    clip1 = VideoFileClip('project_video.mp4')
    project_clip = clip1.fl_image(processing)

    project_output = 'output_project.mp4'
    project_clip.write_videofile(project_output, audio=False)