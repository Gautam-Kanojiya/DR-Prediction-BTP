import { Box,   colors, Typography, List, ListItem, ListItemIcon, ListItemText } from "@mui/material";
import { FiberManualRecord as BulletIcon } from '@mui/icons-material';
import React from "react";
import { Link } from "react-router-dom";
import { tokens } from "../../theme";
import { useTheme } from "@mui/material";
//import FromBox from '../FormBox.jsx';
// import Header from "../../components/Header";
import PersonAddIcon from "@mui/icons-material/PersonAdd";
import TrafficIcon from "@mui/icons-material/Traffic";
import LocalPhoneIcon from '@mui/icons-material/LocalPhone';
import PlaceIcon from '@mui/icons-material/Place';
import WorkOutlineIcon from '@mui/icons-material/WorkOutline';
import Header from "../../components/Header";


function Home(props) {
    const theme = useTheme();
    const colors = tokens(theme.palette.mode);
    const Symptoms = [
      'Spots or dark strings floating in your vision (floaters)',
      'Blurred vision',
      'Fluctuating vision',
      'Dark or empty areas in your vision',
      'Vision loss',
    ];
    const Causes = [
      'To much Sugar in Blood',
      'Blockage of the tiny blood vessels',
      'As a result, the eye attempts to grow new blood vessels. But these new blood vessels do not develop properly and can leak easily.',
      
    ];
    const RistFactors = [
      'Having diabetes for a long time',
      'Poor control of your blood sugar level',
      'High blood pressure',
      'High cholesterol',
      'Pregnancy',
      'Tobacco use',
  
    ];
    const Complications = [
      'Vitreous hemorrhage',
      'Retinal detachment',
      'Glaucoma',
      
  
    ];
  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center">
          <Header title="" subtitle="" />
          <Box 
            display="flex" 
            alignItems="center"
            justifyContent="right"  
            marginTop="0px"  
            
          >
            {/* <Box  margin="20px" >
            <h1>
            <Link to="/login">Login</Link>
                </h1>
            </Box>
            
            <Box margin="20px" >
                <h1>
                <Link to="/signup" fontColor="blue">Signup</Link>
                </h1>
              
            </Box> */}
      </Box>
          
        </Box>
      
      {/* GRID & CHARTS */}
      <Box
          display="grid"
          gridTemplateColumns="repeat(12, 1fr)"
          gridAutoRows="140px"
          gap="20px"
          ml="15px"
        >
          
          {/* ROW 1 */}
          <Box
            gridColumn="span 8"
            gridRow="span 1"
            backgroundColor={colors.primary[400]}
            p="20px"
            //mb="20px"
          >
            <Typography
            variant="h5"
            >
            Diabetic retinopathy is a diabetes complication that affects eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).
At first, diabetic retinopathy might cause no symptoms or only mild vision problems. But it can lead to blindness.</Typography>


          </Box>
          <Box
            gridColumn="span 4"
            gridRow="span 2"
            //backgroundColor={colors.primary[400]}
            //p="10px"
            //mb="20px"
            //alignItems="right"
          >
              <Box display="flex" justifyContent="center" alignItems="center">
                <img
                  alt="profile-user"
                  width="80%"
                  height="210px"
                  src={`../../assets/DR_01.jpeg`}
                  style={{ cursor: "pointer", borderRadius: "5%" }}
                />
              </Box>
              {/* <Typography variant="h4" fontWeight="600" marginTop="3px">
                Vishu Aasliya
              </Typography> */}

          </Box>
          
          {/* Row 2 */}
          <Box
            gridColumn="span 4"
            gridRow="span 2"
            backgroundColor={colors.primary[400]}
            p="10px"
            // mb="20px"
          >
            <Typography 
              sx={{ color: colors.greenAccent[500] }}
              variant="h3">
              Symptoms
            </Typography>
            <List>
              {Symptoms.map((point, index) => (
                <ListItem key={index}>
                  <ListItemIcon >
                    <BulletIcon />
                  </ListItemIcon>
                  <ListItemText primary={point} />
                </ListItem>
              ))}
            </List>

          </Box>

          <Box
            gridColumn="span 4"
            gridRow="span 2"
            backgroundColor={colors.primary[400]}
            p="10px"
            // mb="20px"
          >
            <Typography 
              sx={{ color: colors.greenAccent[500] }}
              variant="h3">
              Causes
            </Typography>
            <List>
              {Causes.map((point, index) => (
                <ListItem key={index}>
                  <ListItemIcon >
                    <BulletIcon />
                  </ListItemIcon>
                  <ListItemText primary={point} />
                </ListItem>
              ))}
            </List>

          </Box>
            
          <Box
            gridColumn="span 4"
            gridRow="span 1"
            // backgroundColor={colors.primary[400]}
            // p="10px"
            mt="-100px"
            //alignItems="right"
          >
              <Box display="flex" justifyContent="center" alignItems="center">
                <img
                  alt="profile-user"
                  width="80%"
                  height="100%"
                  src={`../../assets/DR_mech.jpg`}
                  style={{ cursor: "pointer", borderRadius: "5%" }}
                />
              </Box>
              {/* <Typography variant="h4" fontWeight="600" marginTop="3px">
                Vishu Aasliya
              </Typography> */}

          </Box>

          <Box
            gridColumn="span 4"
            gridRow="span 2"
            backgroundColor={colors.primary[400]}
            p="10px"
            // mb="20px"
          >
            <Typography 
              sx={{ color: colors.greenAccent[500] }}
              variant="h3"
              mb="-10px"
              >
              Risk Factors
            </Typography>
            <List mt="-10px">
              {RistFactors.map((point, index) => (
                <ListItem key={index} >
                  <ListItemIcon >
                    <BulletIcon />
                  </ListItemIcon>
                  <ListItemText primary={point} />
                </ListItem>
              ))}
            </List>

          </Box>  

          <Box
            gridColumn="span 8"
            gridRow="span 2"
            backgroundColor={colors.primary[400]}
            p="10px"
            // mb="20px"
          >
            <Typography 
              sx={{ color: colors.greenAccent[500] }}
              variant="h3"
              mb="-10px"
              >
              Risk Factors
            </Typography>
            <List mt="-10px">
              {Complications.map((point, index) => (
                <ListItem key={index} >
                  <ListItemIcon >
                    <BulletIcon />
                  </ListItemIcon>
                  <ListItemText primary={point} variant="h3"/>
                </ListItem>
              ))}
            </List>

          </Box>  

        </Box>
      {/* <img
                  alt="profile-user"
                  width="50%"
                  height="50%"
                  src={`../../assets/SVG_logo.png`}
                  style={{ cursor: "pointer", borderRadius: "10%" }}
                /> */}
     {/* <h2>{props.name ? `Welcome - ${props.name}` : "Login please"}</h2> */}
    </Box>
  );
}

export default Home;