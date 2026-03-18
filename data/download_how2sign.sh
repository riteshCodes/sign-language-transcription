#!/bin/bash
#
# This script will create the folder structure and download the How2Sign dataset.
# For any questions about the following instructions or the data please contact: amanda.duarte[at]upc.edu
#
# To use this script, first choose the modalities that you would like to download and pass it as an argument to the command.
# For example, to download the "rgb_front_videos", the "rgb_side_videos" and the "english_translation_re-aligned" you can use the following command:
#
# ./download_how2sign.sh rgb_front_videos rgb_side_videos english_translation_re-aligned
#
# The names of the modalities avaliable for download can be found at the botton of this document
################################################################################

# Provide at least one argument to script
if [ $# -lt 1 ]; then
    echo "USAGE: $0 <argument1> <argument2> ..."
    exit
fi

echo "Downloading the How2Sign dataset"

#############################################
# Create folder structure and download data #
#############################################

DOWNLOAD_SCRIPT="./download_script.sh"


check_quota_exceeded() {
    # Use a timeout to prevent long grep operations on large files
    if timeout 10 grep -q "Google Drive - Quota exceeded" "$1"; then
        echo "Quota exceeded error detected in $1."
        return 1 # Indicates error
    elif [ $? -eq 124 ]; then
        return 0 # Assume no error if timeout occurs
    fi
    return 0 # No error
}

# Download file if not exists or if error detected
download_file() {
    local url=$1
    local path=$2

    # Loop until the file is downloaded without errors
    while true; do
        # Check if the file exists
        if [ -f "$path" ]; then
            echo "File $path already exists, checking content..."
            # Check for Google Drive quota exceeded error
            check_quota_exceeded "$path"
            quota_exceeded=$?
            if [ $quota_exceeded -eq 0 ]; then
                echo "File $path is free of errors."
                break # Exit loop if no error
            else
                echo "Redownloading $path due to error detection."
            fi
        else
            echo "File $path does not exist, initiating download..."
        fi

        # If file doesn't exist or error detected, download
        echo "Downloading $path..."
        sh $DOWNLOAD_SCRIPT "$url" "$path"

        # Check again if newly downloaded file has error
        check_quota_exceeded "$path"
        quota_exceeded=$?
        if [ $quota_exceeded -eq 0 ]; then
            echo "File $path downloaded successfully and verified."
            break # Exit loop if file is correctly downloaded
        else
            echo "Error detected post-download, retrying..."
            sleep 1800 # Optional: Sleep to prevent hammering the server too quickly
        fi
    done
}


#------------------------- Green Screen RGB videos - Frontal View -------------------------#
rgb_front_videos() {
    mkdir -p "./How2Sign/video_level/train/rgb_front"
    mkdir -p "./How2Sign/video_level/val/rgb_front"
    mkdir -p "./How2Sign/video_level/test/rgb_front"

    echo "***** Downloading Green Screen RGB videos (Frontal View)... You can go get a coffee, this might take a while!*****"

    ## Train
    ### train_raw_videos.z01
    download_file 'https://docs.google.com/uc?export=download&id=1xWlMM2O3Gbp_8LK5FefoH0TVEmae6jIf' 'train_raw_videos.z01'

	### train_raw_videos.z02
    download_file 'https://docs.google.com/uc?export=download&id=1krtYdpK_LQFgEUCnHxoYAW7EyhLMLWq0' 'train_raw_videos.z02'

	### train_raw_videos.z03
    download_file 'https://docs.google.com/uc?export=download&id=1fXpWRNFhpuVm3ym7lT9vF_bnDjHkvP_K' 'train_raw_videos.z03'

	### train_raw_videos.z04
    download_file 'https://docs.google.com/uc?export=download&id=1IFetFt4AzsxNCMZ0VVpX7YRgFAm58X48' 'train_raw_videos.z04'

	### train_raw_videos.z05
    download_file 'https://docs.google.com/uc?export=download&id=1ZHuuun6Ae-AOLBns3LmuH7w8C9YCB4gH' 'train_raw_videos.z05'

	### train_raw_videos.z06
    download_file 'https://docs.google.com/uc?export=download&id=1FQQIPblk-oLH_vu7h2tDO0oJaZ3xkp5N' 'train_raw_videos.z06'

	### train_raw_videos.z07
    download_file 'https://docs.google.com/uc?export=download&id=19XNgERcolGAMPPgX-Gx_GebSTx3W4o0r' 'train_raw_videos.z07'

	### train_raw_videos.z08
    download_file 'https://docs.google.com/uc?export=download&id=1YN-SA9uzrogEdKeT6UdQUIcuGEyYJILg' 'train_raw_videos.z08'

	### train_raw_videos.z09
    download_file 'https://docs.google.com/uc?export=download&id=1SZQ2GzPLCkRqvsImAjULAPBiuAKi9DE9' 'train_raw_videos.z09'

	### train_raw_videos.zip
    download_file 'https://docs.google.com/uc?export=download&id=1Xe1T5okJiopMXUiH3sc0mdCWNDYSBopd' 'train_raw_videos.zip'

    ## Val
    ### val_raw_videos.zip
    download_file 'https://docs.google.com/uc?export=download&id=1fCkyuKSsc7gauljuL9sx_jBomf3N6i0g' 'val_raw_videos.zip'

    ## Test
    ### test_raw_videos.zip
    download_file 'https://docs.google.com/uc?export=download&id=1z0i6BBGHQ12ChY63hZH56QnczvQ0JfTb' 'test_raw_videos.zip'

    # Merge all train zip files
    echo "***** Preparing the downloaded files... this might take some time! *****"
    cat train_raw_videos.z* > train_raw_videos_all.zip

    unzip train_raw_videos_all.zip -d ./How2Sign/video_level/train/rgb_front && rm -rf train_raw_videos_all.zip
    unzip val_raw_videos.zip   -d ./How2Sign/video_level/val/rgb_front && rm -rf val_raw_videos.zip
    unzip test_raw_videos.zip  -d ./How2Sign/video_level/test/rgb_front && rm -rf test_raw_videos.zip
}

#------------------------- Green Screen RGB videos - Side View -------------------------#
rgb_side_videos()
{
	mkdir -p "./How2Sign/video_level/train/rgb_side"
	mkdir -p "./How2Sign/video_level/val/rgb_side"
	mkdir -p "./How2Sign/video_level/test/rgb_side"

	echo "***** Downloading Green Screen RGB videos (Side View)... This might take a while! *****"

	## Train
	### train_side_raw_videos.z01
	download_file 'https://docs.google.com/uc?export=download&id=1Rmf6LfNWn6lWkAz6Iuj5pMOI2I5p4j1U' 'train_side_raw_videos.z01'

	### train_side_raw_videos.z02
	download_file 'https://docs.google.com/uc?export=download&id=1FytIYIRYrBgAeNWIAhO5vnI2mYOvYC9i' 'train_side_raw_videos.z02'

	### train_side_raw_videos.z03
	download_file 'https://docs.google.com/uc?export=download&id=1kC24jgNgjYYiIYhCRE-gGR28H_2xBBbP' 'train_side_raw_videos.z03'

	### train_side_raw_videos.z04
	download_file 'https://docs.google.com/uc?export=download&id=1JunkM-ImFYao_MwDW9zeqe-6Th6rOLhR' 'train_side_raw_videos.z04'

	### train_side_raw_videos.z05
	download_file 'https://docs.google.com/uc?export=download&id=1-vMckelz9fy4GVNYXRCcy7cJ12X4P3KZ' 'train_side_raw_videos.z05'

	### train_side_raw_videos.z06
	download_file 'https://docs.google.com/uc?export=download&id=1uV413eKsihkNzquN2bwtIQG-OZZMz6sh' 'train_side_raw_videos.z06'

	### train_side_raw_videos.z07
	download_file 'https://docs.google.com/uc?export=download&id=1sU8xrneFJHBzT_PFz4iRPqI8A7HGilhW' 'train_side_raw_videos.z07'

	### train_side_raw_videos.z08
	download_file 'https://docs.google.com/uc?export=download&id=1RPLxeZ54uSZUJSXdPFhXOgeIXziOwTW9' 'train_side_raw_videos.z08'

	### train_side_raw_videos.z09
	download_file 'https://docs.google.com/uc?export=download&id=1tClhr98PszBvFpo9ELKuhbTZZgTGGQqh' 'train_side_raw_videos.z09'

	### train_side_raw_videos.zip
	download_file 'https://docs.google.com/uc?export=download&id=10xrXWgH7iW3E6sgJZDPRwlIhIaDLfHQm' 'train_side_raw_videos.zip'

	## Val
	### val_rgb_side_raw_videos.zip
	download_file 'https://docs.google.com/uc?export=download&id=1Z2H96JT68o7eTChEXPI9z3xyx7zUJPl5' 'val_rgb_side_raw_videos.zip'

	## Test
	### test_rgb_side_raw_videos.zip
	download_file 'https://docs.google.com/uc?export=download&id=1tCQ8KIuuiirXHsh29w0XAMNB3HLIGqgA' 'test_rgb_side_raw_videos.zip'

	# Merge all train zip files
	echo "***** Preparing the downloaded files... this might take some time! *****"
	cat train_side_raw_videos.z* > train_side_raw_videos.zip

	unzip train_side_raw_videos.zip -d ./How2Sign/video_level/train/rgb_side && rm -rf train_side_raw_videos.zip
	unzip val_rgb_side_raw_videos.zip   -d ./How2Sign/video_level/val/rgb_side && rm -rf val_rgb_side_raw_videos.zip
	unzip test_rgb_side_raw_videos.zip  -d ./How2Sign/video_level/test/rgb_side && rm -rf test_rgb_side_raw_videos.zip
}

#------------------------- Green Screen RGB clips -- Frontal view -------------------------#
rgb_front_clips()
{
	mkdir -p "./How2Sign/sentence_level/train/rgb_front"
	mkdir -p "./How2Sign/sentence_level/val/rgb_front"
	mkdir -p "./How2Sign/sentence_level/test/rgb_front"

	echo "***** Downloading and preparing the Green Screen RGB clips (Frontal view) videos *****"

	## Train
	### train_rgb_front_clips.zip
	download_file 'https://docs.google.com/uc?export=download&id=1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW' 'train_rgb_front_clips.zip'

	## Val
	### val_rgb_front_clips.zip
	download_file 'https://docs.google.com/uc?export=download&id=1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO' 'val_rgb_front_clips.zip'


	## Test
	### test_rgb_front_clips.zip
	download_file 'https://docs.google.com/uc?export=download&id=1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG' 'test_rgb_front_clips.zip'

	unzip train_rgb_front_clips.zip -d ./How2Sign/sentence_level/train/rgb_front && rm -rf train_rgb_front_clips.zip
	unzip val_rgb_front_clips.zip   -d ./How2Sign/sentence_level/val/rgb_front && rm -rf val_rgb_front_clips.zip
	unzip test_rgb_front_clips.zip  -d ./How2Sign/sentence_level/test/rgb_front && rm -rf test_rgb_front_clips.zip
}

#-------------------------  Green Screen RGB clips -- Side view -------------------------#
rgb_side_clips()
{
	mkdir -p "./How2Sign/sentence_level/train/rgb_side"
	mkdir -p "./How2Sign/sentence_level/val/rgb_side"
	mkdir -p "./How2Sign/sentence_level/test/rgb_side"

	echo "***** Downloading and preparing the Green Screen RGB clips (Side view) videos *****"

	## Train
	### train_rgb_side_clips.zip
	download_file 'https://docs.google.com/uc?export=download&id=1oiw861NGp4CKKFO3iuHGSCgTyQ-DXHW7' 'train_rgb_side_clips.zip'

	## Val
	### val_rgb_side_clips.zip
	download_file 'https://docs.google.com/uc?export=download&id=1mxL7kJPNUzJ6zoaqJyxF1Krnjo4F-eQG' 'val_rgb_side_clips.zip'

	## Test
	### test_rgb_side_clips.zip
	download_file 'https://docs.google.com/uc?export=download&id=1j9v9P7UdMJ0_FVWg8H95cqx4DMSsrdbH' 'test_rgb_side_clips.zip'


	unzip train_rgb_side_clips.zip -d ./How2Sign/sentence_level/train/rgb_side && rm -rf train_rgb_side_clips.zip
	unzip val_rgb_side_clips.zip   -d ./How2Sign/sentence_level/val/rgb_side && rm -rf val_rgb_side_clips.zip
	unzip test_rgb_side_clips.zip  -d ./How2Sign/sentence_level/test/rgb_side && rm -rf test_rgb_side_clips.zip
}

#------------------------- B-F-H 2D Keypoints clips -- Frontal view -------------------------#
rgb_front_2D_keypoints()
{
	mkdir -p "./How2Sign/sentence_level/train/rgb_front/features"
	mkdir -p "./How2Sign/sentence_level/val/rgb_front/features"
	mkdir -p "./How2Sign/sentence_level/test/rgb_front/features"

	echo "***** Downloading B-F-H 2D Keypoints clips (Frontal view) files... This might take a while! *****"
	## Train
	### train_2D_keypoints.tar.gz
	download_file 'https://docs.google.com/uc?export=download&id=1TBX7hLraMiiLucknM1mhblNVomO9-Y0r' 'train_2D_keypoints.tar.gz'

	## Val
	### val_2D_keypoints.tar.gz
	download_file 'https://docs.google.com/uc?export=download&id=1JmEsU0GYUD5iVdefMOZpeWa_iYnmK_7w' 'val_2D_keypoints.tar.gz'

	## Test
	### test_2D_keypoints.tar.gz
	download_file 'https://docs.google.com/uc?export=download&id=1g8tzzW5BNPzHXlamuMQOvdwlHRa-29Vp' 'test_2D_keypoints.tar.gz'

	echo "***** Preparing the downloaded files... this might take some time! *****"
	tar -xf train_2D_keypoints.tar.gz -C ./How2Sign/sentence_level/train/rgb_front/features && rm -rf train_2D_keypoints.tar.gz
	tar -xf val_2D_keypoints.tar.gz   -C ./How2Sign/sentence_level/val/rgb_front/features && rm -rf val_2D_keypoints.tar.gz
	tar -xf test_2D_keypoints.tar.gz  -C ./How2Sign/sentence_level/test/rgb_front/features && rm -rf test_2D_keypoints.tar.gz
}

# # B-F-H 2D Keypoints clips -- Side view
# rgb_side_2D_keypoints()
# {
# 	echo "Creating B-F-H 2D Keypoints clips -- Side view folders"
# 	mkdir -p "./How2Sign/sentence_level/train/rgb_side/features/openpose_output"
# 	mkdir -p "./How2Sign/sentence_level/val/rgb_side/features/openpose_output"
# 	mkdir -p "./How2Sign/sentence_level/test/rgb_side/features/openpose_output"

# 	unzip train_rgb_side_2D_keypoints.zip -d ./How2Sign/sentence_level/train/rgb_side/features
# 	unzip val_rgb_side_2D_keypoints.zip   -d ./How2Sign/sentence_level/val/rgb_side/features
# 	unzip test_rgb_side_2D_keypoints.zip  -d ./How2Sign/sentence_level/test/rgb_side/features
# }

#------------------------- English Translation -------------------------#
english_translation()
{
	mkdir -p "./How2Sign/sentence_level/train/text/en/raw_text"
	mkdir -p "./How2Sign/sentence_level/val/text/en/raw_text"
	mkdir -p "./How2Sign/sentence_level/test/text/en/raw_text"

	echo "***** Downloading and preparing the English Translation text files *****"
	## Train
	### how2sign_train.csv
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lq7ksWeD3FzaIwowRbe_BvCmSmOG12-f' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lq7ksWeD3FzaIwowRbe_BvCmSmOG12-f" -O  how2sign_train.csv && rm -rf /tmp/cookies.txt

	## Val
	### how2sign_val.csv
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aBQUClTlZB504JtDISJ0DJlbuYUZCGu3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aBQUClTlZB504JtDISJ0DJlbuYUZCGu3" -O  how2sign_val.csv && rm -rf /tmp/cookies.txt

	## Test
	### how2sign_test.csv
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ScxYnEjILZMn22qKjQj8Wyr_F0nha7kG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ScxYnEjILZMn22qKjQj8Wyr_F0nha7kG" -O  how2sign_test.csv && rm -rf /tmp/cookies.txt

	mv how2sign_train.csv How2Sign/sentence_level/train/text/en/raw_text
	mv how2sign_val.csv How2Sign/sentence_level/val/text/en/raw_text
	mv how2sign_test.csv How2Sign/sentence_level/test/text/en/raw_text
}

#------------------------- English Translation re-aligned -------------------------#
english_translation_re-aligned()
{
	mkdir -p "./How2Sign/sentence_level/train/text/en/raw_text/re_aligned"
	mkdir -p "./How2Sign/sentence_level/val/text/en/raw_text/re_aligned"
	mkdir -p "./How2Sign/sentence_level/test/text/en/raw_text/re_aligned"

	echo "***** Downloading and preparing the re-aligned English Translation text files *****"
	## Train
	### how2sign_realigned_train.csv
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dUHSoefk9OxKJnHrHPX--I4tpm9QD0ok' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dUHSoefk9OxKJnHrHPX--I4tpm9QD0ok" -O  how2sign_realigned_train.csv && rm -rf /tmp/cookies.txt

	## Val
	### how2sign_realigned_val.csv
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI" -O  how2sign_realigned_val.csv && rm -rf /tmp/cookies.txt

	## Test
	### how2sign_realigned_test.csv
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu" -O  how2sign_realigned_test.csv && rm -rf /tmp/cookies.txt

	mv how2sign_realigned_train.csv How2Sign/sentence_level/train/text/en/raw_text/re_aligned
	mv how2sign_realigned_val.csv How2Sign/sentence_level/val/text/en/raw_text/re_aligned
	mv how2sign_realigned_test.csv How2Sign/sentence_level/test/text/en/raw_text/re_aligned
}

## TODO
# Gloss annotations
# Panoptic Studio data

# Modalities avaliable for download
for ARG in "$@"
do
	shift
	case "${ARG}" in
		"rgb_front_videos") 		rgb_front_videos;;
		"rgb_side_videos")		rgb_side_videos;;
		"rgb_front_clips")		rgb_front_clips;;
		"rgb_side_clips")		rgb_side_clips;;
		"rgb_front_2D_keypoints")	rgb_front_2D_keypoints;;
		# "rgb_side_2D_keypoints")	rgb_side_2D_keypoints;;
		"english_translation")	english_translation;;
		"english_translation_re-aligned")	english_translation_re-aligned;;
		*)				echo "${ARG}: Invalid argument given";;
	esac
	echo "Thank you for downloading the How2Sign dataset. Please check the README file for information about the files you just downloaded and feel free to contact us if you have any questions."
done
#
################################################################################