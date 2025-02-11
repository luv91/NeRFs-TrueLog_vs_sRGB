#!/bin/bash

# List of directories
directories=(
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010636_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010637_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/llff_truelog/GX010636_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010639_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010640_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/llff_truelog/GX010640_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results/llff_truelog/GX010405_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010404_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010405_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results/llff_truelog/GX010416_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010416_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010417_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010396_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010397_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results/llff_truelog/GX010396_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010619_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010620_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/llff_truelog/GX010620_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010624_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010625_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/llff_truelog/GX010625_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010628_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010629_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/llff_truelog/GX010629_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010630_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/l_360/GX010631_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results360/llff_truelog/GX010631_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results/llff_lr1eminus2to1eminus3_redo/GX010454_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results/llff_truelog/GX010455_LLFF_Processed"
    "/work/SuperResolutionData/LuvNeRF/results/llff_lr1eminus2to1eminus3/GX010455_LLFF_Processed"
)

for dir in "${directories[@]}"; do
    render_dir="$dir/render"
    
    if [ ! -d "$render_dir" ]; then
        echo "Render directory does not exist: $render_dir"
        continue
    fi

    subdirs=($(find "$render_dir" -mindepth 1 -maxdepth 1 -type d))
    
    for subdir in "${subdirs[@]}"; do
        for tag in "color" "distance_mean" "distance_median"; do
            output_video="$subdir/new_360_${tag}_video.mp4"

            # Collect all image files for the tag
            frames=($(find "$subdir" -type f -name "${tag}_*.tiff" | sort))
            frame_count="${#frames[@]}"
            
            if [ "$frame_count" -eq 0 ]; then
                echo "No frames found in $subdir for tag $tag"
                continue
            fi

            # Create a temporary folder
            temp_dir=$(mktemp -d)
            
            # Calculate step size for 360 frames
            if [ "$frame_count" -lt 360 ]; then
                echo "Interpolating $frame_count frames to 360 frames in $subdir for tag $tag"
                
                # Copy existing frames with appropriate spacing
                for ((i = 0; i < 360; i++)); do
                    # Calculate which original frame to use
                    original_frame_index=$(( (i * frame_count) / 360 ))
                    # Create symbolic link with new name
                    ln -s "${frames[$original_frame_index]}" "$temp_dir/frame_$(printf "%03d" $i).png"
                done
            else
                # If we have more than 360 frames, use first 360
                for i in {0..359}; do
                    ln -s "${frames[i]}" "$temp_dir/frame_$(printf "%03d" $i).png"
                done
            fi

            # Generate the 360° video with 6s duration (60fps * 6s = 360 frames)
            ffmpeg -framerate 60 -i "$temp_dir/frame_%03d.png" \
                   -c:v libx264 -preset slow -crf 22 \
                   -r 60 -pix_fmt yuv420p \
                   "$output_video"
            
            echo "Saved video to $output_video"

            # Clean up temporary directory
            rm -r "$temp_dir"
        done
    done
done