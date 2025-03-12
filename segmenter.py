import os
import json
import argparse
import logging

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | l: %(lineno)s | %(message)s",
    handlers=[logging.FileHandler("logs/log.log", mode="w"), logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentator for the images")

    parser.add_argument(
        "--input_img_dir",
        "-i",
        type=str,
        default="data/jpg",
        help="Path to the directory containing the images",
        required=True,
    )
    parser.add_argument(
        "--output_img_dir",
        "-o",
        type=str,
        default="data/jpg",
        help="Path to the directory containing the images",
        required=True,
    )

    return parser.parse_args()


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()


class VideoSegmenter:
    """Simple class to segment set of continuos images from the 3D reconstructor from inintial input"""

    def __init__(
        self,
        model_cfg: str,
        sam2_checkpoint: str,
        input_img_dir: str,
        output_img_dir: str,
    ):
        self.logger = LOGGER
        self.device = self._get_device()

        self.predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=self.device
        )
        # Prompts
        self.bbox = []  # Bounding box
        self.points = []  # Points
        self.pt_labels = []  # Point labels

        self.mask2show = None
        self.mask = None
        self.inference_state = None
        self._is_prepared = False  # If the initial selection and mask run was done

        self.selecter_mode = "bbox"

        self.input_img_dir = input_img_dir
        self.output_img_dir = output_img_dir
        os.makedirs(self.output_img_dir, exist_ok=True)
        self.logger.info(f"Input images directory: {self.input_img_dir}")
        self.logger.info(f"Output images directory: {self.output_img_dir}")

    def _get_device(self):
        """Get the device to use for the model based on the available hardware

        Returns:
            torch.device: Available device
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.logger.info(f"Using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            self.logger.warning(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        return device

    def _extract_coordinates(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Get the first point of the bounding box
            if self.selecter_mode == "bbox":
                self.bbox = [x, y]
            elif self.selecter_mode == "point_include":
                self.points.append([x, y])
                self.pt_labels.append(1)
                self.logger.info(f"Selected include point: {x, y}")
            elif self.selecter_mode == "point_exclude":
                self.points.append([x, y])
                self.pt_labels.append(0)
                self.logger.info(f"Selected exclude point: {x, y}")

        elif event == cv2.EVENT_LBUTTONUP:
            if self.selecter_mode == "bbox":
                self.bbox.extend([x, y])  # Add the second point to the bounding box
                self.logger.info(f"Current bounding box: {self.bbox}")
                self.mask2show = None
                self.mask = None

    def _visualize_masks(self, mask, image):
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        unique_values = np.unique(mask)

        for u_val in unique_values:
            # print(u_val)
            if u_val == 0:  # Skip 0 signifying background
                continue
            color = tuple(np.random.randint(0, 255, 3, dtype=np.uint8).tolist())
            # print(color)
            color = [255, 255, 255]

            cur_mask = np.array(mask == u_val)
            cur_mask = cur_mask.astype(np.uint8)
            cur_mask = np.repeat(cur_mask[:, :, np.newaxis], 3, axis=2)
            cur_mask = np.array(color)[None, None, :] * cur_mask
            cur_mask = cur_mask.astype(np.uint8)

            weighted = cv2.addWeighted(img, 0.5, cur_mask, 0.5, 0)

            img[mask == u_val] = weighted[mask == u_val]

        cv2.imshow("Mask", img)
        self.mask2show = img

    def _selection_help(self) -> None:
        """Writes the description of the selection process to the logger"""
        self.logger.info("Press 'q' to quit")
        self.logger.info("Press 'r' to reset the selection")
        self.logger.info("Press 'e' to switch prompting to exclude point")
        self.logger.info("Press 'i' to switch prompting to include point")
        self.logger.info("Press 'b' to switch prompting to selecting bounding box")
        self.logger.info(
            "Press 'm' to mask the image, needs to be runned once before the Enter"
        )
        self.logger.info("Press 'Enter' to confirm the selection")

    def select_init(self, image):
        """Select the initial point to start the segmentation

        Args:
            image (np.ndarray): Image to segment

        Returns:
            np.ndarray: Selected point
        """
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("Select the initial point")
        cv2.setMouseCallback(
            "Select the initial point", self._extract_coordinates, param=None
        )
        self.logger.info("Select the bounding box to start the segmentation")
        self._selection_help()
        frame_vis = image_bgr.copy()
        ret = None
        mask_color = np.array([255, 255, 255])

        while True:
            # >>> VISUALIZATION >>>
            frame_vis = image_bgr.copy()
            # Mask display without the darkening
            if self.mask is not None:
                # color = [255, 255, 255]
                cur_mask = np.array(self.mask == 1)
                cur_mask = np.repeat(cur_mask[:, :, np.newaxis], 3, axis=2)
                cur_mask = mask_color[None, None, :] * cur_mask
                cur_mask = cur_mask.astype(np.uint8)
                weighted = cv2.addWeighted(image_bgr, 0.5, cur_mask, 0.5, 0)
                frame_vis[self.mask == 1] = weighted[self.mask == 1]

            # Draw the bounding box
            if len(self.bbox) == 4:
                cv2.rectangle(
                    frame_vis,
                    (self.bbox[0], self.bbox[1]),
                    (self.bbox[2], self.bbox[3]),
                    (0, 255, 0),
                    2,
                )

            # Points
            for pt, label in zip(self.points, self.pt_labels):
                if label == 1:
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 0, 255)  # Red
                cv2.circle(frame_vis, tuple(pt), 5, color, -1)

            cv2.imshow("Select the initial point", frame_vis)
            # <<< VISUALIZATION <<<

            key = cv2.waitKey(1) & 0xFF
            if key == ord("h"):  # Help
                self._selection_help()

            # NOTE: Commented out now as the selection is required to continue 
            # if key == ord("q"):
            #      
            #     break

            
            elif key == ord("r"):  # Reset prompts
                self.logger.info("Resetting the bounding box")
                self.bbox = []
                self.points = []
                self.pt_labels = []
                self.mask2show = None
                self.mask = None
                self._is_prepared = False

            elif key == ord("e"):  # Exclude point
                self.selecter_mode = "point_exclude"
                self.logger.info("Selector mode: Exclude point")
            elif key == ord("i"):  # Include point
                self.selecter_mode = "point_include"
                self.logger.info("Selector mode: Include point")
            elif key == ord("b"):  # Bounding box
                self.selecter_mode = "bbox"
                self.logger.info("Selector mode: Bounding box")

            elif key == ord("m"):  # Mask the image
                # TODO: Write another checks for validitz of the promprs atleast one
                if len(self.points) != len(self.pt_labels):
                    self.logger.warning("Points not defined; try again")
                    continue

                if len(self.bbox) == 4:
                    _, out_obj_ids, out_mask_logits = (
                        self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=0,
                            obj_id=1,
                            box=np.array(self.bbox),
                            points=np.array(self.points, dtype=np.float32),
                            labels=np.array(self.pt_labels, dtype=np.int32),
                        )
                    )
                    self._is_prepared = True
                elif (
                    len(self.points) > 0 and sum(self.pt_labels) > 0
                ):  # Points are defined and at least one include point is seledted
                    _, out_obj_ids, out_mask_logits = (
                        self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=0,
                            obj_id=1,
                            points=np.array(self.points, dtype=np.float32),
                            labels=np.array(self.pt_labels, dtype=np.int32),
                        )
                    )
                    self._is_prepared = True
                else:
                    self.logger.warning(
                        "Bounding box or include points not defined; please try again"
                    )
                    continue

                # sorted_ind = np.argsort(scores)[::-1]
                # masks = masks[sorted_ind]
                # scores = scores[sorted_ind]
                # logits = logits[sorted_ind]
                # mask = masks[0, :, :] * 1
                # all_masks = self._combine_mask(mask, img_name, img.shape[:2])
                self.mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                # self._visualize_masks(self.mask, image)

            elif key == 13:  # Enter returns the selected bounding box
                if not self._is_prepared:
                    self.logger.warning(
                        "Initial mask not prepared, press 'm' to prepare the mask"
                    )
                    continue
                break

        cv2.destroyAllWindows()
        return ret

    def run_segmentation(self) -> None:
        """Run the segmentation on the images in the directory

        Args:
            img_dir (str): Path to the directory containing the images
        """
        video_segments = {}
        video_names = sorted(os.listdir(self.input_img_dir))
        video_names = [name for name in video_names if name.endswith(".jpg")]

        first_img = cv2.imread(
            os.path.join(self.input_img_dir, video_names[0])
        )  # Loads the first image in BGR format
        first_img = cv2.cvtColor(
            first_img, cv2.COLOR_BGR2RGB
        )  # SAM2 expects RGB images

        inference_state = segmenter.predictor.init_state(video_path=self.input_img_dir)
        segmenter.predictor.reset_state(inference_state)

        segmenter.inference_state = inference_state

        # Waits untill the bounding box is selected
        bbox = segmenter.select_init(first_img)

        video_segments = (
            {}
        )  # video_segments contains the per-frame segmentation results
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in segmenter.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # print(video_segments.keys())
        cv2.namedWindow("masked")
        # os.makedirs(os.path.join(self.output_img_dir, "weighted"), exist_ok=True)
        # TODO: Maybe add mask overide to regenerate the mask, if some unwanted mask is generated
        for image_name in video_names:
            image = cv2.imread(os.path.join(self.input_img_dir, image_name))
            image_id = int(image_name.split(".")[0])
            mask = video_segments[image_id][1][0] * 255

            # print(mask.shape, mask.dtype)
            # print(image.shape, image.dtype)

            fin_image = image.copy()
            fin_image = cv2.cvtColor(fin_image, cv2.COLOR_BGR2BGRA)
            fin_image[mask == 0] = 0
            fin_image[:, :, 3] = mask

            # cv2.imshow("maskedf", fin_image)
            cv2.imwrite(
                os.path.join(self.output_img_dir, image_name.replace(".jpg", ".png")),
                fin_image,
            )

            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            vis_img = image.copy()
            weighted = cv2.addWeighted(vis_img, 0.5, mask, 0.5, 0)

            # cv2.imwrite(os.path.join(self.output_img_dir, "weighted", image_name.replace(".jpg", ".png")), weighted)

            # put images side by side into one image
            together = np.hstack((weighted, fin_image[:, :, :3]))
            cv2.imshow("masked", together)

            key = cv2.waitKey(1) & 0xFF

        self.logger.info(f"Segmentation saved to {self.output_img_dir}")

    def run_bbox_extacrtion(self) -> None:
        """Run the segmentation on the images in the directory

        Args:
            img_dir (str): Path to the directory containing the images
        """
        bboxes = {}
        video_segments = {}
        video_names = sorted(os.listdir(self.input_img_dir))
        video_names = [name for name in video_names if name.endswith(".jpg")]

        first_img = cv2.imread(
            os.path.join(self.input_img_dir, video_names[0])
        )  # Loads the first image in BGR format
        first_img = cv2.cvtColor(
            first_img, cv2.COLOR_BGR2RGB
        )  # SAM2 expects RGB images

        inference_state = segmenter.predictor.init_state(video_path=self.input_img_dir)
        segmenter.predictor.reset_state(inference_state)

        segmenter.inference_state = inference_state

        # Waits untill the bounding box is selected
        bbox = segmenter.select_init(first_img)

        video_segments = (
            {}
        )  # video_segments contains the per-frame segmentation results
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in segmenter.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # print(video_segments.keys())
        cv2.namedWindow("masked")
        os.makedirs(os.path.join(self.output_img_dir, "mask"), exist_ok=True)
        # TODO: Maybe add mask overide to regenerate the mask, if some unwanted mask is generated
        for image_name in tqdm(video_names, desc="Postprocessing images"):
            image = cv2.imread(os.path.join(self.input_img_dir, image_name))
            image_id = int(image_name.split(".")[0])
            mask = video_segments[image_id][1][0] * 255
            # cnt = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # bbox_from_mask = cv2.boundingRect(cnt[0])
            # bboxes[image_id] = bbox_from_mask

            # print(mask.shape, mask.dtype)
            # print(image.shape, image.dtype)

            fin_image = image.copy()
            fin_image = cv2.cvtColor(fin_image, cv2.COLOR_BGR2BGRA)
            fin_image[mask == 0] = 0
            fin_image[:, :, 3] = mask

            # cv2.imshow("maskedf", fin_image)
            cv2.imwrite(
                os.path.join(self.output_img_dir, image_name.replace(".jpg", ".png")),
                fin_image,
            )

            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            vis_img = image.copy()
            weighted = cv2.addWeighted(vis_img, 0.5, mask, 0.5, 0)

            cv2.imwrite(
                os.path.join(
                    self.output_img_dir, "mask", image_name.replace(".jpg", ".png")
                ),
                mask,
            )

            # put images side by side into one image
            together = np.hstack((weighted, fin_image[:, :, :3]))
            cv2.imshow("masked", together)

            key = cv2.waitKey(1) & 0xFF

        self.logger.info(f"Segmentation saved to {self.output_img_dir}")
        with open(os.path.join(self.output_img_dir, "bboxes.json"), "w") as f:
            json.dump(bboxes, f)


if __name__ == "__main__":
    # EXAMPLE COMMAND
    # "/home/testbed/miniconda3/envs/SAM2/bin/python /home/testbed/Projects/segment-anything-2/segmenter.py -i data/jpg -o data/masked_out"

    args = parse_args()
    input_img_dir = args.input_img_dir
    output_img_dir = args.output_img_dir

    sam2_checkpoint = "/home/testbed/Projects/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"  # HAVE TO BE DIRECT PATH NOT RELATIVE
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    segmenter = VideoSegmenter(
        model_cfg, sam2_checkpoint, input_img_dir, output_img_dir
    )

    # segmenter.run_segmentation()
    segmenter.run_bbox_extacrtion()
