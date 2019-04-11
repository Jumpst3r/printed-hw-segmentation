import json
import sys
import time

import requests

url = "http://134.21.131.72:8080/algorithms"


# The request body for creating the method on DIVAServices
# This contains all required information about:
#   - general information about the method
#   - the parameters the method needs
#   - the ouputs the method generates
#   - which Docker Image to use and what script to execute
payload = "{\
            \"general\": \
                {   \"name\": \"Nicolas Printed Handwritten Segmentation\",\
                    \"description\": \"pixelwise printed and handwritten segmentation\",\
                    \"developer\": \"Nicolas Dutly\",\
                    \"affiliation\": \"University of Fribourg\",\
                    \"email\": \"nicolas.dutly@unifr.ch\",\
                    \"author\": \"Nicolas Dutly\",\
                    \"type\": \"segmentation\",\
                    \"license\": \"MIT\",\
                    \"ownsCopyright\": \"1\"\
                },\
            \"input\": [\
                {\"file\":\
                    {   \"name\": \"inputImage\",\
                        \"description\": \"The input image to segment\",\
                        \"options\": {\
                            \"required\": true,\
                            \"mimeTypes\": {\
                                \"allowed\": [\"image/jpeg\",\"image/png\"],\
                                \"default\": \"image/jpeg\"\
                                }\
                            }\
                        }\
                    },{\
                    \"outputFolder\": {}}\
            ],\
            \"output\": [\
                {\"file\": {\
                    \"name\": \"segoutput\",\
                    \"type\": \"image\",\
                    \"description\": \"Generated Segmented Image\",\
                    \"options\": {\
                        \"mimeTypes\": {\
                            \"allowed\": [\"image/jpeg\",\"image/png\",\"image/tiff\"],\
                            \"default\": \"image/jpeg\"\
                        },\
                        \"colorspace\": \"rgb\",\
                        \"visualization\": true}\
                        }\
                    }\
                ],\
            \"method\": {\
                \"imageType\": \"docker\",\
                \"imageName\": \"jumpst3r/printed-hw-segmentation\",\
                \"testData\": \"https://drive.google.com/uc?authuser=0&id=1UAa6rWknEkEAb-aoxxxLSOJ5jMzi535J&export=download\",\
                \"executableType\": \"bash\",\
                \"executable_path\": \"/input/script.sh\"\
            }\
        }"

headers = {'content-type': "application/json"}

#start the installation process on DIVAServices
response = requests.request("POST", url, data=payload, headers=headers)
response = json.loads(response.text)

identifier = response['identifier']
status_link = "http://134.21.72.190:8080/algorithms/" + identifier
print(response)
# check the current status of the installation
status_response = json.loads(requests.request("GET", status_link).text)
status_code = status_response['status']['statusCode']
while(status_code != 200):
    #quit the program if something went wrong
    if(status_code == 500):
        sys.stderr.write('Error in creating the method on DIVAServices')
        sys.exit()
    time.sleep(1)
    status_response = json.loads(requests.request("GET", status_link).text)
    status_code = status_response['status']['statusCode']
    print('current status: ' + str(status_code) + ' current message: ' + status_response['status']['statusMessage'])

print("Sucessfully created method on DIVAServices")
