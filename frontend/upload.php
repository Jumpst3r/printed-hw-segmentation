<?php
ini_set('max_execution_time', 300);
$target_dir = "uploads/";
$target_file = $target_dir . basename($_FILES["fileToUpload"]["name"]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));
// Check if image file is a actual image or fake image
if(isset($_POST["submit"])) {
    $check = getimagesize($_FILES["fileToUpload"]["tmp_name"]);
    if($check !== false) {
        echo "File is an image - " . $check["mime"] . ".";
        $uploadOk = 1;
    } else {
        echo "File is not an image.";
        $uploadOk = 0;
    }
}

// Check file size
if ($_FILES["fileToUpload"]["size"] > 50000000000000) {
    echo "Sorry, your file is too large.";
    $uploadOk = 0;
}
// Allow certain file formats
if($imageFileType != "jpg" && $imageFileType != "png" && $imageFileType != "jpeg"
&& $imageFileType != "gif" ) {
    echo "Sorry, only JPG, JPEG, PNG & GIF files are allowed.";
    $uploadOk = 0;
}
// Check if $uploadOk is set to 0 by an error
if ($uploadOk == 0) {
    echo "Sorry, your file was not uploaded.";
// if everything is ok, try to upload file
} else {
    if (move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $target_file)) {
        echo "The file ". basename( $_FILES["fileToUpload"]["name"]). " has been uploaded.\n\n";
        echo "Uploading on DIVA services...";

        $cmd = 'python executeOnDivaservices.py ' . $_FILES["fileToUpload"]["name"] . ' http://' . '134.21.133.202' . '/uploads/' . $_FILES["fileToUpload"]["name"] . " output/" ;

        $curl = curl_init();

        curl_setopt_array($curl, array(
          CURLOPT_PORT => "8080",
          CURLOPT_URL => "http://134.21.72.190:8080/collections",
          CURLOPT_RETURNTRANSFER => true,
          CURLOPT_ENCODING => "",
          CURLOPT_MAXREDIRS => 10,
          CURLOPT_TIMEOUT => 30,
          CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
          CURLOPT_CUSTOMREQUEST => "POST",
          CURLOPT_POSTFIELDS => "{\n \"files\":[\n  {\n   \"type\":\"url\",\n   \"value\":\"http://134.21.131.89/uploads/".$_FILES["fileToUpload"]["name"]."\"\n  }\n  ]\n}",
          CURLOPT_HTTPHEADER => array(
            "content-type: application/json"
          ),
        ));

        $response = curl_exec($curl);
        $err = curl_error($curl);

        curl_close($curl);

        if ($err) {
          echo "cURL Error #:" . $err;
        } else {
                echo 'uploaded image to DIVA server, executing method..\n';
                $someObject = json_decode($response, true);
                $im_id = $someObject["collection"] ."/".$_FILES["fileToUpload"]["name"];
                echo "image identifier: ".$im_id;
                sleep(1);
                $curl = curl_init();
                curl_setopt_array($curl, array(
                  CURLOPT_PORT => "8080",
                  CURLOPT_URL => "http://134.21.72.190:8080/segmentation/nicolasprintedhandwrittensegmentation/1",
                  CURLOPT_RETURNTRANSFER => true,
                  CURLOPT_ENCODING => "",
                  CURLOPT_MAXREDIRS => 10,
                  CURLOPT_TIMEOUT => 30,
                  CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
                  CURLOPT_CUSTOMREQUEST => "POST",
                  CURLOPT_POSTFIELDS => "{\n \"data\":[\n  {\n   \"inputImage\": \"".$im_id."\"\n  }\n ]\n}",
                  CURLOPT_HTTPHEADER => array(
                    "content-type: application/json"
                  ),
                ));

                $response = curl_exec($curl);
                $err = curl_error($curl);

                curl_close($curl);

                if ($err) {
                  echo "cURL Error #:" . $err;
                } else {
                  $res = json_decode($response, true);
                  $res_link = $res['results'][0]['resultLink'];
                  
                  $curl = curl_init();

                curl_setopt_array($curl, array(
                  CURLOPT_PORT => "8080",
                  CURLOPT_URL => $res_link,
                  CURLOPT_RETURNTRANSFER => true,
                  CURLOPT_ENCODING => "",
                  CURLOPT_MAXREDIRS => 10,
                  CURLOPT_TIMEOUT => 30,
                  CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
                  CURLOPT_CUSTOMREQUEST => "GET",
                  CURLOPT_POSTFIELDS => "",
                  CURLOPT_HTTPHEADER => array(
                    "content-type: application/json"
                  ),
                ));

                $response = curl_exec($curl);
                $err = curl_error($curl);

                if ($err) {
                  echo "cURL Error #:" . $err;
                } else {
                   $res = json_decode($response, true);
                   while($res['status'] == 'running'){
                       sleep(2);
                       $response = curl_exec($curl);
                        $res = json_decode($response, true);
                   }
                   echo $res['output'][0]['file']['url'];
                   
                   $img = 'output/res.png';  
  
                // Function to write image into file 
                file_put_contents($img, file_get_contents($res['output'][0]['file']['url']));
                echo "DOOONE";
                   
                }
                  
                }
        }
}
}
?>