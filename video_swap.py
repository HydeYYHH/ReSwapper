import os
import argparse
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm

# Import necessary modules from ReSwapper
from insightface.app import FaceAnalysis
import face_align
import Image
from StyleTransferModel_128 import StyleTransferModel

# Initialize face analysis
faceAnalysis = FaceAnalysis(name='buffalo_l')
faceAnalysis.prepare(ctx_id=0, det_size=(512, 512))

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    device = get_device()
    model = StyleTransferModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model

def swap_face(model, target_face, source_face_latent):
    device = get_device()

    target_tensor = torch.from_numpy(target_face).to(device)
    source_tensor = torch.from_numpy(source_face_latent).to(device)

    with torch.no_grad():
        swapped_tensor = model(target_tensor, source_tensor)
    
    swapped_face = Image.postprocess_face(swapped_tensor)
    
    return swapped_face, swapped_tensor

def create_target(target_image, resolution):
    target_face = faceAnalysis.get(target_image)[0]
    aligned_target_face, M = face_align.norm_crop2(target_image, target_face.kps, resolution)
    target_face_blob = Image.getBlob(aligned_target_face, (resolution, resolution))

    return target_face_blob, M

def create_source(source_img_path):
    source_image = cv2.imread(source_img_path)

    faces = faceAnalysis.get(source_image)
    if len(faces) == 0:
        raise Exception("No face detected in source image")

    source_face = faces[0]
    source_latent = Image.getLatent(source_face)

    return source_latent

def process_video(args):
    # Load model
    model = load_model(args.modelPath)
    
    # Load source face latent
    source_latent = create_source(args.source)
    
    # Handle face attribute modification if specified
    if args.face_attribute_direction is not None:
        direction = np.load(args.face_attribute_direction)
        direction = direction / np.linalg.norm(direction)
        source_latent += direction * args.face_attribute_steps
    
    # Open video file
    video_capture = cv2.VideoCapture(args.target)
    
    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if needed
    output_folder = os.path.dirname(args.outputPath)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
    out = cv2.VideoWriter(args.outputPath, fourcc, fps, (width, height))
    
    print(f"Processing video: {args.target}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Process each frame
    start_time = time.time()
    
    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        # Read frame
        ret, frame = video_capture.read()
        if not ret:
            break
        
        try:
            # Detect faces in the frame
            faces = faceAnalysis.get(frame)
            if len(faces) > 0:
                # For now, we'll process only the first detected face
                # Create target face blob
                target_face_blob, M = create_target(frame, args.resolution)
                
                # Perform face swap
                swapped_face, _ = swap_face(model, target_face_blob, source_latent)
                
                # Paste the swapped face back to the original frame
                if not args.no_paste_back:
                    frame = Image.blend_swapped_image(swapped_face, frame, M)
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            # Continue with the original frame if there's an error
            pass
        
        # Write the processed frame
        out.write(frame)
    
    # Release resources
    video_capture.release()
    out.release()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Video processing completed!")
    print(f"Output saved to: {args.outputPath}")
    print(f"Processing time: {processing_time:.2f} seconds ({total_frames/processing_time:.2f} FPS)")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Video Face Swapping with ReSwapper')
    
    parser.add_argument('--target', required=True, help='Target video path')
    parser.add_argument('--source', required=True, help='Source image path')
    parser.add_argument('--outputPath', required=True, help='Output video path')
    parser.add_argument('--modelPath', required=True, help='Model path')
    parser.add_argument('--no-paste-back', action='store_true', help='Disable pasting back the swapped face onto the original frame')
    parser.add_argument('--resolution', type=int, default=128, help='Resolution')
    parser.add_argument('--face_attribute_direction', default=None, help='Path of direction.npy')
    parser.add_argument('--face_attribute_steps', type=float, default=0, help='Face attribute adjustment steps')

    return parser.parse_args()

def main():
    args = parse_arguments()
    process_video(args)

if __name__ == "__main__":
    main()