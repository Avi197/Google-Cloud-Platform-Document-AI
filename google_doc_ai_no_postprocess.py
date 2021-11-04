import math
import os.path
import re
from glob import glob
from timeit import default_timer as timer

import cv2
import img2pdf
import numpy as np
import pytesseract
from deskew import determine_skew
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from pdf2image import convert_from_path

KEY_VALUE_THRESHOLD = 0.5
TABLE_THRESHOLD = 0.5

project_id = 'ocr-document-317803'
location = 'us'  # Format is 'us' or 'eu'
processor_id = 'bc01f356d8ca7a13'  # Create processor in Cloud Console
bucket_name = 'vnd_ai_ocr_doc'

opts = {}
if location == "eu":
    opts = {"api_endpoint": "eu-documentai.googleapis.com"}

client = documentai.DocumentProcessorServiceClient(client_options=opts)


class DeskewException(Exception):
    pass


mime_type = "application/pdf"


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    if not storage.Blob(bucket=bucket, name=destination_blob_name).exists(storage_client):
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to {destination_blob_name}.")
    else:
        print(f"File {destination_blob_name} exist.")


def rotate_2(image, angle, background):
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def deskew(image_np):
    img_cv_grey = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(img_cv_grey)
    if angle < -45.0:
        angle += 90.0
    if abs(angle) >= 0.1:
        rotated = rotate_2(image_np, angle, (0, 0, 0))
        return rotated.astype(np.uint8)
    else:
        return image_np.astype(np.uint8)


def teserract_deskew(image):
    try:
        osd = pytesseract.image_to_osd(image)
        # using regex we search for the angle(in string format) of the text
        angle = re.search('(?<=Rotate: )\d+', osd).group(0)
        if (angle == '0'):
            image = deskew(image)
            return image
        elif (angle == '90'):
            image = rotate_2(image, 270, (0, 0, 0)).astype(np.uint8)  # rotate(image,angle,background_color)
            image = deskew(image)
            return image
        elif (angle == '180'):
            image = rotate_2(image, 180, (0, 0, 0)).astype(np.uint8)
            image = deskew(image)
            return image
        elif (angle == '270'):
            image = rotate_2(image, 90, (0, 0, 0)).astype(np.uint8)
            image = deskew(image)
            return image
    except DeskewException:
        print('Error image: No information like empty page. Skipping ...')
        return image


def deskew_pdf(in_path, out_path, temp_path="./temp/temp_deskew"):
    # Split pdf into page
    # print("deskewing file ")
    pages = convert_from_path(in_path, 155)
    deskew_pages = []
    for index, page in enumerate(pages):
        # print(f"deskewing page {index}")
        filename = f"{temp_path}/{index}.jpg"
        deskew_pages.append(filename)

        image = np.array(page)
        image = teserract_deskew(image)
        cv2.imwrite(filename, image)

    # Merge images to pdf
    with open(out_path, "wb") as f:
        f.write(img2pdf.convert(deskew_pages))


def get_document_ai_result(file_path):
    with open(file_path, "rb") as f:
        image_content = f.read()
    document = {"content": image_content, "mime_type": mime_type}

    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    # Configure the process request
    request = {"name": name, "raw_document": document}

    # Recognizes text entities in the PDF document
    result = client.process_document(request=request)
    document = result.document
    return document.pages


# this function return pages output from doc ai
def get_document_ai_batch_result(input_uri, output_uri, file_name, timeout=3000):
    gcs_input_uri = f"gs://{bucket_name}/{input_uri}/{file_name}"
    gcs_output_uri = "gs://{}/{}/{}".format(bucket_name, output_uri, file_name[:-4])

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob_list = list(bucket.list_blobs(prefix="{}/{}".format(output_uri, file_name[:-4])))

    if len(blob_list) == 0:
        gcs_documents = documentai.GcsDocuments(
            documents=[{"gcs_uri": gcs_input_uri, "mime_type": mime_type}]
        )

        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
        output_config = documentai.DocumentOutputConfig(
            gcs_output_config={"gcs_uri": gcs_output_uri})

        name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

        request = documentai.types.document_processor_service.BatchProcessRequest(
            name=name,
            input_documents=input_config,
            document_output_config=output_config,
        )

        operation = client.batch_process_documents(request)
        operation.result(timeout=timeout)
        blob_list = list(bucket.list_blobs(prefix="{}/{}".format(output_uri, file_name[:-4])))
    else:
        print("File {}/{} exist.".format(output_uri, file_name[:-4]))

    pages = []
    for i, blob in enumerate(blob_list):
        # If JSON file, download the contents of this blob as a bytes object.
        if ".json" in blob.name:
            blob_as_bytes = blob.download_as_bytes()
            document = documentai.types.Document.from_json(blob_as_bytes)
            print(f"Fetched file {i + 1}")
            pages.extend(document.pages)
        else:
            print(f"Skipping non-supported file type {blob.name}")
    return pages


# this function return list of json output from doc ai
def get_document_ai_batch_result_json(input_uri, output_uri, file_name, timeout=3000):
    gcs_input_uri = f"gs://{bucket_name}/{input_uri}/{file_name}"
    gcs_output_uri = f"gs://{bucket_name}/{output_uri}/{file_name}"
    # gcs_output_uri = "gs://{}/{}/{}".format(bucket_name, output_uri, file_name)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob_list = list(bucket.list_blobs(prefix="{}/{}".format(output_uri, file_name)))

    if len(blob_list) == 0:
        gcs_documents = documentai.GcsDocuments(
            documents=[{"gcs_uri": gcs_input_uri, "mime_type": mime_type}]
        )

        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
        output_config = documentai.DocumentOutputConfig(
            gcs_output_config={"gcs_uri": gcs_output_uri})

        name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

        request = documentai.types.document_processor_service.BatchProcessRequest(
            name=name,
            input_documents=input_config,
            document_output_config=output_config,
        )

        operation = client.batch_process_documents(request)
        operation.result(timeout=timeout)
        blob_list = list(bucket.list_blobs(prefix="{}/{}".format(output_uri, file_name)))
    else:
        print("File {}/{} exist.".format(output_uri, file_name))

    return blob_list


# this function call doc ai api to process the pdf, but return nothing
# the purpose is just to call this function multiple time at once and let doc ai process (async)
def async_document_ai_process(input_uri, output_uri, file_name):
    file_name = file_name.replace('.pdf', '')
    gcs_input_uri = f"gs://{bucket_name}/{input_uri}/{file_name}.pdf"
    gcs_output_uri = f"gs://{bucket_name}/{output_uri}/{file_name}"
    # storage_client = storage.Client()
    # bucket = storage_client.get_bucket(bucket_name)

    gcs_documents = documentai.GcsDocuments(
        documents=[{"gcs_uri": gcs_input_uri, "mime_type": mime_type}]
    )

    input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
    output_config = documentai.DocumentOutputConfig(
        gcs_output_config={"gcs_uri": gcs_output_uri})

    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    request = documentai.types.document_processor_service.BatchProcessRequest(
        name=name,
        input_documents=input_config,
        document_output_config=output_config,
    )

    operation = client.batch_process_documents(request)

    operation.add_done_callback(process)


def process(future):
    result = future.result()


def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:
        print(blob.name)


def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    blob_list = []
    storage_client = storage.Client()

    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    for blob in blobs:
        blob_list.append(blob.name)

    if delimiter:
        blob_list = []
        for prefix in blobs.prefixes:
            blob_list.append(prefix)

    return blob_list


def check_blob_exists(bucket_name, name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if storage.Blob(bucket=bucket, name=name).exists(storage_client):
        return True
    else:
        return False


# check for file that not in "in" folder and "out" folder
def check_new_file(in_path, out_path):
    in_li = list_blobs_with_prefix(bucket_name, in_path, delimiter=None)
    out_li = list_blobs_with_prefix(bucket_name, out_path, delimiter='/')

    in_list = [i.replace(f'{in_path}/', '').replace('.pdf', '') for i in in_li[1:]]
    out_list = [i.replace(f'{out_path}/', '').replace('/', '') for i in out_li]

    return list(set(in_list).difference(out_list))


# check for file that not in "in" folder and "out" folder for dummy files
def check_new_file_dummy(in_path, out_path):
    in_li = list_blobs_with_prefix(bucket_name, in_path, delimiter=None)
    out_li = list_blobs_with_prefix(bucket_name, out_path, delimiter=None)

    in_list = [i.replace(f'{in_path}/', '').replace('.pdf', '') for i in in_li[1:]]
    out_list = [i.replace(f'{out_path}/', '').replace('.pdf', '') for i in out_li[1:]]

    return list(set(in_list).difference(out_list))


if __name__ == '__main__':

    # TODO: trong thư mục "in" không có nhưng "out" lại có

    # NOTE:
    # create these folders on bucket
    # main folder: gcp_process
    # sub folder:
    # - pdf_in: pdf upload location
    # - deskewed_in: deskewed pdf file to feed to doc ai
    # - deskewed_out: json output from doc ai (not needed anymore, preplace by json_out)
    # - json_out: json output from doc ai
    # - excel_out: final output in excel format

    # init path variable
    # ---------------------------------------------------------------------------------------
    temp_folder = './temp'

    # temp folder to store original pdf file on VM
    pdf_path_local = f"{temp_folder}/pdf"

    # temp folder for deskewing process (contain jpg files)
    deskewed_pdf_temp = f"{temp_folder}/temp_deskew"

    # temp folder to store the deskewed file on VM
    deskewed_path_local = f"{temp_folder}/deskewed_pdf"

    # temp folder to store xlsx output file on VM
    xlsx_path_local = f"{temp_folder}/xlsx"

    # temp folder to store dummy file on VM
    dummy_temp = f"{temp_folder}/dummy"

    # pdf folder on bucket
    in_pdf_bucket = "gcp_process/pdf_in"

    # deskew_in: location of the deskewed files
    deskewed_path_bucket_full = f"gs://{bucket_name}/gcp_process/deskewed_in"
    deskewed_path_bucket = "gcp_process/deskewed_in"

    # deskew_out: location of the deskewed files' output json
    deskewed_json_path_bucket_full = f"gs://{bucket_name}/gcp_process/json_out"
    deskewed_json_path_bucket = "gcp_process/json_out"

    # dummy_out: create a dummy for stopping the new file checking progress
    dummy_path_bucket_full = f"gs://{bucket_name}/gcp_process/dummy_out"
    dummy_path_bucket = "gcp_process/dummy_out"

    # location of xlsx output on bucket
    xlsx_path_bucket_full = f"gs://{bucket_name}gcp_process/excel_out"
    xlsx_path_bucket = "gcp_process/excel_out"
    # ---------------------------------------------------------------------------------------

    # create temp folders if not exists on VM
    # ---------------------------------------------------------------------------------------
    if not os.path.exists(pdf_path_local):
        os.makedirs(pdf_path_local)
    if not os.path.exists(deskewed_pdf_temp):
        os.makedirs(deskewed_pdf_temp)
    if not os.path.exists(deskewed_path_local):
        os.makedirs(deskewed_path_local)
    if not os.path.exists(xlsx_path_local):
        os.makedirs(xlsx_path_local)
    if not os.path.exists(dummy_temp):
        os.makedirs(dummy_temp)
    # ---------------------------------------------------------------------------------------
    print("Starting the process")
    while True:

        list_new_file = check_new_file_dummy(in_pdf_bucket, dummy_path_bucket)

        # check for files that are not processed by doc ai and return a list of those files
        if list_new_file:
            print("Found new file")
            # print(list_new_file)
            # print('---------------------------------------------------------------------')

            file_paths = []
            for file_name in list_new_file:
                file_name_pdf = f'{file_name}.pdf'
                # the list of file_name are not contain .pdf extension, so it's needed to add .pdf at the end
                file_paths.append(
                    (file_name_pdf, f'{pdf_path_local}/{file_name_pdf}', f'{deskewed_path_local}/{file_name_pdf}'))

            # start deskew and doc ai process on VM
            # ---------------------------------------------------------------------------------------

            # get the file path from file_paths, not from init variable
            # careful to not mistake those
            for (file_name_pdf, pdf_file_path, deskewed_file_path) in file_paths:

                #  print(f"Process for {file_name_pdf}")

                file_name_raw = file_name_pdf.replace('.pdf', '')
                print(f"Process for {file_name}")
                # if file is deskewed already, skip deskew part
                if not check_blob_exists(bucket_name, f'{deskewed_path_bucket}/{file_name_pdf}'):

                    # only download if the file hasn't been deskewed before to prevent taking VM storage
                    print("downloading file to VM")
                    download_blob(bucket_name, f"{in_pdf_bucket}/{file_name_pdf}", f"{pdf_path_local}/{file_name_pdf}")

                    start_time = timer()
                    # deskew và trả ra pdf tương ứng ban đầu
                    deskew_pdf(pdf_file_path, deskewed_file_path, deskewed_pdf_temp)
                    end_time = timer()
                    print(f"deskew took {end_time - start_time} seconds to complete")

                    # upload deskewed pdf to bucket to process
                    upload_blob(bucket_name, deskewed_file_path, f"{deskewed_path_bucket}/{file_name_pdf}")

                    # Remove temp files
                    # ---------------------------------------------------------------------------------------
                    filelist = glob(f"{deskewed_pdf_temp}/*.jpg")
                    for f in filelist:
                        os.remove(f)

                    # delete the folder using init variable, not from file_paths
                    # TODO: pretty stupid here, fix later
                    filelist = glob(f"{pdf_path_local}/*")
                    for f in filelist:
                        os.remove(f)

                    # delete the folder using init variable, not from file_paths
                    # TODO: pretty stupid here, fix later
                    filelist = glob(f"{deskewed_path_local}/*")
                    for f in filelist:
                        os.remove(f)
                else:
                    print("file already deskewed")

                print("----------------------------------------------------------------------------")

                # ---------------------------------------------------------------------------------------

                # list of doc ai json output for that pdf
                # use get_doc..._json for the blob_list not the other one

                # fuck knows why the fucking naming of gcp bucket is pure magic shit
                # if the folder doesn't have .pdf extension, the process wont even run and return 400 error
                async_document_ai_process(deskewed_path_bucket, deskewed_json_path_bucket, file_name_pdf)

                # create a dummy file and upload to bucket as a temporary check for doc ai processing file
                with open(f'{dummy_temp}/{file_name_pdf}', 'w') as fp:
                    pass
                upload_blob(bucket_name, f'{dummy_temp}/{file_name_pdf}', f"{dummy_path_bucket}/{file_name_pdf}")

                print("running the process in the background")

                filelist = glob(f"{dummy_temp}/*")
                for f in filelist:
                    os.remove(f)

            print("Waiting for new file")