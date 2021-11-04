# Google-Cloud-Platform-Document-AI
Interact with GCP Document AI's API


## Steps:
- Send a pdf document to GCP bucket
- Check for new pdf file on GCP bucket
- Check if file is read by Doc AI
- Call Doc AI if file hasn't been read
- Return Doc AI json result to GCP bucket

Both scripts can handle multiple request send to Document AI API using google api's long-running operation  

#### google_doc_ai:
- Process json result on GCP Virtual machine
- Run on a GCP Virtual machine
- Extract information in json result and upload final result to GCP bucket
- Much faster to process 
- Cost extra for GCP Virtual machine
#### google_doc_ai_no_postprocess: 
- Process json result on local, after download json result from GCP bucket
- Run on local
- Download and extract information in json result and return final result on local machine
- Slower since it run on local machine or directly on client machine
- No extra cost, only Document AI

