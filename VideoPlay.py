import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from preprocessing_module import *
from calibration_module import *
from sliding_window import *


def draw_on_original(undist, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane with low confidence region in red
    cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0))

    # confidence region in green
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
    # to select whether diagnostic video(1) or submission video(0)
    verbose = 0
    # undistor the image
    undist = get_camera_calibration(img)
    # apply the thresholding operation
    thresh_combined, grad_th, col_th = thresholding(undist)
    # Perspective transformation
    perspective, Minv = perspective_transform(thresh_combined)
    perspective = cv2.cvtColor(perspective, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    # pass the perspective image to the lane fitting stage
    slides_pers, left_fitx, right_fitx, ploty, avg_cur, dist_centre_val = sliding_window(perspective)
    # draw the detected lanes on the original image
    mapped_lane = draw_on_original(undist, left_fitx, right_fitx, ploty, Minv)
    # font and text for drawing the offset and curvature
    curvature = "Estimated lane curvature %.2fm" % (avg_cur)
    dist_centre = "Estimated offset from lane center %.2fm" % (dist_centre_val)
    font = cv2.FONT_HERSHEY_COMPLEX
    # using cv2 for drawing text/images in diagnostic pipeline.
    if verbose == 1:
        middlepanel = np.zeros((120, 900, 3), dtype=np.uint8)
        l1 = np.zeros((50, 50, 3), dtype=np.uint8)
        l2 = np.zeros((50, 50, 3), dtype=np.uint8)
        l3 = np.zeros((50, 50, 3), dtype=np.uint8)
        l4 = np.zeros((50, 50, 3), dtype=np.uint8)
        l5 = np.zeros((50, 50, 3), dtype=np.uint8)
        l6 = np.zeros((50, 50, 3), dtype=np.uint8)
        l7 = np.zeros((50, 50, 3), dtype=np.uint8)
        legend = np.zeros((240, 1200, 3), dtype=np.uint8)

        cv2.putText(middlepanel, curvature, (30, 60), font, 1, (255, 255, 255), 2)
        cv2.putText(middlepanel, dist_centre, (30, 90), font, 1, (255, 255, 255), 2)
        cv2.putText(l1, "1", (15, 35), font, 1, (255, 255, 0), 2)
        cv2.putText(l2, "2", (15, 30), font, 1, (255, 255, 0), 2)
        cv2.putText(l3, "3", (15, 30), font, 1, (255, 255, 0), 2)
        cv2.putText(l4, "4", (15, 30), font, 1, (255, 255, 0), 2)
        cv2.putText(l5, "5", (15, 30), font, 1, (255, 255, 0), 2)
        cv2.putText(l6, "6", (15, 30), font, 1, (255, 255, 0), 2)
        cv2.putText(l7, "7", (15, 30), font, 1, (255, 255, 0), 2)
        text = "1-Detected Lanes, 2-Color Threshold\n3-Gradient Threshold, 4-Thresholding operations combined\n" \
               "5-Perspective Transformation, 6-Original Frame\n" \
               "7-Mapping Polynomials, Blue line-current frame polynomial fit,\n" \
               "Green line-smoothened polynomial fit, Pink - Lane pixels"

        y0, dy = 50, 40
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv2.putText(legend, line, (50, y), font, 1, (255, 255, 255), 2)

        diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # 2
        diagScreen[0:360, 1200:1560] = cv2.resize(np.dstack((col_th * 255, col_th * 255, col_th * 255)), (360, 360),
                                                  interpolation=cv2.INTER_AREA)
        # 3
        diagScreen[0:360, 1560:1920] = cv2.resize(np.dstack((grad_th * 255, grad_th * 255, grad_th * 255)), (360, 360),
                                                  interpolation=cv2.INTER_AREA)
        # 4
        diagScreen[360:720, 1200:1560] = cv2.resize(thresh_combined * 255, (360, 360), interpolation=cv2.INTER_AREA)
        # 5
        diagScreen[360:720, 1560:1920] = cv2.resize(
            np.dstack((perspective * 255, perspective * 255, perspective * 255)), (360, 360),
            interpolation=cv2.INTER_AREA)
        # 7
        diagScreen[720:1080, 1560:1920] = cv2.resize(slides_pers, (360, 360), interpolation=cv2.INTER_AREA)
        # 6
        diagScreen[720:1080, 1200:1560] = cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA)
        # 1
        diagScreen[0:720, 0:1200] = cv2.resize(mapped_lane, (1200, 720), interpolation=cv2.INTER_AREA)

        # radii,offset and legend here
        diagScreen[720:840, 0:900] = middlepanel
        diagScreen[0:50, 0:50] = l1
        diagScreen[0:50, 1200: 1250] = l2
        diagScreen[0:50, 1560:1610] = l3
        diagScreen[720:770, 1560:1610] = l7
        diagScreen[360:410, 1560:1610] = l5
        diagScreen[720:770, 1200:1250] = l6
        diagScreen[360:410, 1200:1250] = l4
        diagScreen[840:1080, 0:1200] = legend
        # if diagnosis then return this image
        return diagScreen
    # else return the original mapped imaged with the curvature and offset drawn
    cv2.putText(mapped_lane, curvature, (30, 60), font, 1.2, (255, 0, 0), 2)
    cv2.putText(mapped_lane, dist_centre, (30, 120), font, 1.2, (255, 0, 0), 2)
    return mapped_lane

if __name__ == '__main__':

    clip1 = VideoFileClip('project_video.mp4')
    project_clip = clip1.fl_image(processing)

    project_output = 'output_project.mp4'
    project_clip.write_videofile(project_output, audio=False)