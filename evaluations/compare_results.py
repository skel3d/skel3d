import csv
import os
from PIL import Image, ImageDraw


def prepare_input_image(path, row):
    image_path = os.path.join(path, row['id'])
    image = Image.open(image_path)
    # draw = ImageDraw.Draw(image)
    # draw.text((10, 10), row['id'], fill='purple')    
    return image

def prepare_image(path, row, counter_row):
    image_path = os.path.join(path, row['id'])
    image = Image.open(image_path)
    l1loss = float(row['l1loss'])
    ssim = float(row['ssim'])
    psnr = float(row['psnr'])
    lpips = float(row['lpips'])

    counter_l1loss = float(counter_row['l1loss'])
    counter_ssim = float(counter_row['ssim'])
    counter_psnr = float(counter_row['psnr'])
    counter_lpips = float(counter_row['lpips'])

    # Draw the numbers on the corresponding image
    draw = ImageDraw.Draw(image)    
    draw.text((10, 10), f'l1loss: {l1loss:.5f}', fill='green' if l1loss < counter_l1loss else 'red')
    draw.text((10, 30), f'ssim: {ssim:.5f}' , fill='green' if ssim > counter_ssim else 'red')
    draw.text((10, 50), f'psnr: {psnr:.5f}', fill='green' if psnr > counter_psnr else 'red')
    draw.text((10, 70), f'lpips: {lpips:.5f}', fill='green' if lpips < counter_lpips else 'red')
    return image

def compare_csv_files(file1, file2):
    largest_differences = {'l1loss': 0, 'ssim': 0, 'psnr': 0, 'lpips': 0}
    
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reader1 = csv.DictReader(f1)
        reader2 = csv.DictReader(f2)
        
        for row1, row2 in zip(reader1, reader2):
            src_image = prepare_input_image(source_path, row1)
            target_image = prepare_input_image(target_path, row1)
            skel_image = prepare_input_image(skel_path, row1)
            image1 = prepare_image(path1, row1, row2)
            image2 = prepare_image(path2, row2, row1)


            concatenated_image1 = Image.new('RGB', (src_image.width + target_image.width+skel_image.width, max(src_image.height, target_image.height)))
            concatenated_image1.paste(src_image, (0, 0))
            concatenated_image1.paste(target_image, (src_image.width, 0))
            concatenated_image1.paste(skel_image, (src_image.width+target_image.width, 0))

            # Concatenate the images
            concatenated_image2 = Image.new('RGB', (image1.width + image2.width, max(image1.height, image2.height)))
            concatenated_image2.paste(image1, (0, 0))
            concatenated_image2.paste(image2, (image1.width, 0))

            concatenated_image = Image.new('RGB', (concatenated_image1.width + concatenated_image2.width, max(concatenated_image1.height, concatenated_image2.height))) 
            concatenated_image.paste(concatenated_image1, (0, 0))
            concatenated_image.paste(concatenated_image2, (concatenated_image1.width, 0))
            

            # Display the concatenated image
            concatenated_image.save(os.path.join(outpath, row1['id']))
            


file1 = 'skel3d_time.csv'
file2 = 'skel3d_time.csv'
source_path = '/outputs/hdf5_skel3d_time_epoch=000041.ckpt/conditioning/'
target_path = '/outputs/hdf5_skel3d_time_epoch=000041.ckpt/inputs/'
skel_path = '/outputs/hdf5_skel3d_time_epoch=000041.ckpt/timegt/'
path1 = '/outputs/hdf5_skel3d_time_epoch=000041.ckpt/samples_cfg_scale_3.00/'
path2 = '/outputs/hdf5_skel3d_time_epoch=000041.ckpt/samples_cfg_scale_3.00/'


outpath = '/outputs/combined_images/'

os.makedirs(outpath, exist_ok=True)

compare_csv_files(file1, file2)