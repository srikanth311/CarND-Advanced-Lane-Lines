import matplotlib.pyplot as plt
import numpy as np
import cv2

class PlotMyImages(object):
    def plot_my_images(source_image, result_image, title_for_source_image=None, title_for_result_image=None):
        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(source_image)
        ax1.set_title(title_for_source_image, fontsize=40)

        ax2.imshow(result_image)
        ax2.set_title(title_for_result_image, fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    def plot_an_image(image, title_for_the_image=None):
        plt.imshow(image, cmap='gray')
        #plt.imshow(image)
        plt.show()
    
    @staticmethod
    def draw_overlay(all_variables):
        all_variables.color_warp = np.zeros_like(all_variables.undistorted_image, dtype='uint8') 

        pts_left = np.array([np.transpose(np.vstack([all_variables.left_lane_plot_x, all_variables.left_lane_plot_y]))])
        #print(all_variables.left_lane_plot_x.shape)
        #print(all_variables.left_lane_plot_y.shape)

        #print(all_variables.right_lane_plot_x.shape)
        #print(all_variables.right_lane_plot_y.shape)

        pts_right = np.array([np.flipud(np.transpose(np.vstack([all_variables.right_lane_plot_x, all_variables.right_lane_plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(all_variables.color_warp, np.int_([pts]), (0,255, 0))

        all_variables.color_unwarped = cv2.warpPerspective(all_variables.color_warp, all_variables.perspective_inverse_matrix, (all_variables.undistorted_image.shape[1], all_variables.undistorted_image.shape[0]))
        all_variables.img_processed = cv2.addWeighted(all_variables.undistorted_image, 1, all_variables.color_unwarped, 0.3, 0)

        label_str = 'Radius of curvature: %.1f m' % all_variables.radius_of_curvature
        cv2.putText(all_variables.img_processed, label_str, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

        label_str = 'Vehicle offset from lane center: %.1f m' % all_variables.vehicle_offset
        cv2.putText(all_variables.img_processed, label_str, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)

    @staticmethod
    def draw_sliding_windows(all_variables):
        #Draw Sliding windows for left and right path
        img = (np.dstack((all_variables.warped_binary_image, all_variables.warped_binary_image, all_variables.warped_binary_image))*255).astype('uint8')
        img = PlotMyImages.draw_lane_windows(img, all_variables.left_lane_window_positions, all_variables.left_lane_points_x, all_variables.left_lane_points_y, all_variables.left_lane_plot_x, all_variables.left_lane_plot_y, [255,0,0])
        img = PlotMyImages.draw_lane_windows(img, all_variables.right_lane_window_positions, all_variables.right_lane_points_x, all_variables.right_lane_points_y, all_variables.right_lane_plot_x, all_variables.right_lane_plot_y, [255,0,0])

        all_variables.img_sliding_window = img
        #PlotMyImages.plot_an_image(all_variables.img_sliding_window)

    @staticmethod
    def draw_lane_windows(img,window_positions,points_x,points_y,plot_x,plot_y,color):

        img[points_y,points_x] = [255,0,0]

        for wp in window_positions:
            cv2.rectangle(img,(wp[0],wp[1]),(wp[2],wp[3]),(0,255,0), 3)

        poly_pts=np.dstack( (plot_x,plot_y) ).astype(np.int32)
        cv2.polylines(img,poly_pts,False,(255,255,0),4)

        img_overlay = np.zeros_like(img)
        margin = 100
        path_pts1 = np.array([np.transpose(np.vstack([plot_x-margin, plot_y]))])
        path_pts2 = np.array([np.flipud(np.transpose(np.vstack([plot_x+margin, plot_y])))])
        path_pts =  np.hstack((path_pts1, path_pts2)).astype(np.int32)

        cv2.fillPoly(img_overlay, path_pts, (0,255, 0))
        img = cv2.addWeighted(img, 1, img_overlay, 0.3, 0)

        return img
    
    @staticmethod
    def plot_all_images(images, rows, cols, file_name):
        # cv2.imwrite('output_images/imageName.jpg',255*binaryoutput)
        for index in range(0, len(images)):
            cv2.imwrite("output_images/final_output_images/" + images[index][1] + ".png", images[index][0])
            plt.subplot(rows, cols, index+1)
            plt.title(images[index][1])
            plt.tick_params(axis='both', which='major', labelsize=8)
            plt.tick_params(axis='both', which='minor', labelsize=6)
            if(len(images[index])==2):
                plt.imshow(images[index][0])
            else:
                plt.imshow(images[index][0],images[index][2])
        plt.tight_layout()
        plt.savefig(file_name)