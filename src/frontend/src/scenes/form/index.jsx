import React, { useState } from 'react';
import { Box, Button, TextField, IconButton, Typography, useTheme } from "@mui/material";
import axios from 'axios';
import { tokens } from "../../theme";
import { Formik } from "formik";
import * as yup from "yup";
import useMediaQuery from "@mui/material/useMediaQuery";
import Header from "../../components/Header";
import FromBox from '../../components/FormBox';
import DownloadOutlinedIcon from "@mui/icons-material/DownloadOutlined";

const ImageUploader = () => {
  // const [leftEyeImage, setLeftEyeImage] = useState(null);
  // const [rightEyeImage, setRightEyeImage] = useState(null); 
  // new addons
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [responseText, setResponseText] = useState('');

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async (event) => {
    event.preventDefault();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Send the image to the backend API using axios
      const response = await axios.post('http://127.0.0.1:8000/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      console.log(response, "response");
      console.log(response.data, "data");

      // Handle the response from the backend if needed
      setResponseText(response.data);
      console.log('Response from backend:', response);
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  const theme = useTheme();
    const colors = tokens(theme.palette.mode);
  return (
    <Box
    
    >
        <Header title="Test Page" subtitle="Evaluate Diabetic Retinopathy Possibility" />
        <h2>Upload Your Eye Images</h2>
        {/* left eye box */}
        <Box
            gridColumn="span 3"
            backgroundColor={colors.primary[400]}
            display="flex"
            alignItems="center"
            justifyContent="center"
        >
            <FromBox
                title="EYE IMAGE"
                subtitle="Upload High quality images for accurate results"
        
            />
            
            <Box>
                <label>Eye Image:</label>
                <input type="file" accept="image/*" onChange={handleFileChange} />
            </Box>
            
        </Box>
        <br />
        {/* right eye box not in use rn */}
        
        

        <Box 
            display="flex"
            alignItems="center"
            justifyContent="center"
        >
                <Button onClick={handleUpload}
                sx={{
                    backgroundColor: colors.blueAccent[700],
                    color: colors.grey[100],
                    fontSize: "14px",
                    fontWeight: "bold",
                    padding: "10px 20px",
                }}
                >
                <DownloadOutlinedIcon sx={{ mr: "10px" }} />
                Submit
                </Button>
                <p>{uploadStatus}</p>
            </Box>
          
                {/*Result box  */}
        <Box
            gridColumn="span 12"
            backgroundColor={colors.primary[400]}
            display="flex"
            
            // alignItems="center"
            // justifyContent="center"
            // width="90%"
            // height = "100%"
        >   
            
            {responseText && (
                  <Box ml="15px">

                    <Typography
                      variant="h5"
                      color={colors.greenAccent[500]}
                      sx={{ mt: "15px" }}
                    >
                       {responseText.message}
                
                   </Typography>
                
                    <Box
                    
                      display="inline-flex"
                      // width="50%"
                      textAlign="center"
                      alignItems="center"
                      justifyContent="space-between"
                      mr="15px"
                    >
                      <Typography >
                        <Typography
                          variant="h6"
                          color={colors.greenAccent[500]}
                          sx={{ mt: "15px" }}
                        >
                          Predicted Grade
                    
                        </Typography>{responseText.predicted_grade}</Typography>
                      {/* <br />
                      <p >Predicted Value: {responseText.predicted_value}</p> */}
                    </Box>

                    <Box
                    
                      display="inline-flex"
                      // width="50%"
                      textAlign="center"
                      alignItems="left "
                      justifyContent="space-between"
                      mr="15px"
                    >
                      <Typography >
                        <Typography
                          variant="h6"
                          color={colors.greenAccent[500]}
                          sx={{ mt: "15px" }}
                        >
                          
                    
                        </Typography></Typography>
                      {/* <br />
                      <p >Predicted Value: {responseText.predicted_value}</p> */}
                    </Box>
                    
                    {/* Recall */}
                   
                    
                   


                  </Box>)}
            
            <Typography
                variant="h5"
                color={colors.greenAccent[500]}
                sx={{ mt: "15px" }}
              >
                
                
              </Typography>
              
            
        </Box>
    </Box>
  );
};

export default ImageUploader;
