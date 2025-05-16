import { Box, Button, IconButton, Typography, useTheme } from "@mui/material";
import { tokens } from "../../theme";
import { mockTransactions } from "../../data/mockData";
import DownloadOutlinedIcon from "@mui/icons-material/DownloadOutlined";
import EmailIcon from "@mui/icons-material/Email";
import PointOfSaleIcon from "@mui/icons-material/PointOfSale";
import PersonAddIcon from "@mui/icons-material/PersonAdd";
import TrafficIcon from "@mui/icons-material/Traffic";
import LocalPhoneIcon from '@mui/icons-material/LocalPhone';
import PlaceIcon from '@mui/icons-material/Place';
import WorkOutlineIcon from '@mui/icons-material/WorkOutline';
import Header from "../../components/Header";

import StatBox from "../../components/StatBox";


const Profile = (props) => {
    const theme = useTheme();
    const colors = tokens(theme.palette.mode);
  
    return (
      <Box m="20px">
        {/* HEADER */}
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Header title={props.name ? `Welcome ${props.name}` : "WELCOME"} subtitle="" />
           
        </Box>
  
        {/* GRID & CHARTS */}
        <Box
          display="grid"
          gridTemplateColumns="repeat(12, 1fr)"
          gridAutoRows="140px"
          gap="20px"
        >
          
          {/* ROW 1 */}
          <Box
            gridColumn="span 12"
            gridRow="span 2"
            backgroundColor={colors.primary[400]}
            p="30px"
            mb="20px"
          >
            
            <Box
              display="flex"
              flexDirection="column"
              alignItems="left"
              //mt="25px"
            >
              <Box display="flex" justifyContent="left" alignItems="center">
                <img
                  alt="profile-user"
                  width="150px"
                  height="150px"
                  
                  src={`../../assets/user.png`}
                  style={{ cursor: "pointer", borderRadius: "50%" }}
                />
                
                <Box>

                  <Typography variant="h2" fontWeight="600" marginTop="3px" ml="25px">
                    USER NAME
                    </Typography>
                  {theme.palette.mode === "dark" ? (
                    <Typography
                    variant="h4"
                    color={colors.greenAccent[500]}
                    sx={{ mt: "15px",
                          ml:"25px"}}
                  >
                    user@gmail.com
                  </Typography>
                    ) : (
                    <Typography
                        variant="h4"
                        color={colors.greenAccent[200]}
                        sx={{ mt: "15px",
                              ml:"25px" }}
                    >
                        user@gmail.com
                    </Typography>
                    )}
                  <Typography 
                    variant="h4" 
                    sx={{ mt: "15px",
                        ml:"25px" }}> 
                        <LocalPhoneIcon /> +91 123456789
                  </Typography>

                  <Typography 
                    variant="h4" 
                    sx={{ mt: "15px",
                        ml:"25px" }}> 
                        <WorkOutlineIcon /> Occupation
                  </Typography>
                </Box>
              </Box>
              
              <Box display="flex" justifyContent="left" alignItems="center" >
                  
                  
                    <PlaceIcon
                      sx={{ color: colors.greenAccent[600], fontSize: "35px" }}
                    />
                    <Typography 
                      variant="h4" 
                      sx={{ mt: "10px",
                          ml:"5px" }}> 
                         Location...
                    </Typography>
              </Box>
    
            </Box>
  
          </Box>
          

          
            
            
        </Box>

        {/* Reports box starts  */}
        <Box
                gridColumn="span 10"
                gridRow="span 2"
                backgroundColor={colors.primary[400]}
                overflow="auto"
              >
                <Box
                  display="flex"
                  justifyContent="space-between"
                  alignItems="center"
                  borderBottom={`4px solid ${colors.primary[500]}`}
                  colors={colors.grey[100]}
                  p="15px"
                >
                  <Typography color={colors.grey[100]} variant="h5" fontWeight="600">
                    Reports
                  </Typography>
                </Box>
                {mockTransactions.map((transaction, i) => (
                  <Box
                    key={`${transaction.txId}-${i}`}
                    display="flex"
                    justifyContent="space-between"
                    alignItems="center"
                    borderBottom={`4px solid ${colors.primary[500]}`}
                    p="15px"
                  >
                    <Box>
                      <Typography
                        color={colors.greenAccent[500]}
                        variant="h5"
                        fontWeight="600"
                      >
                        ID : {transaction.txId}
                      </Typography>
                      {/* <Typography color={colors.grey[100]}>
                        {transaction.user}
                      </Typography> */}
                    </Box>
                    <Box color={colors.grey[100]}>{transaction.date}</Box>
                    <Box
                      backgroundColor={colors.greenAccent[500]}
                      p="5px 10px"
                      borderRadius="4px"
                    >
                      {transaction.cost}
                    </Box>
                  </Box>
                ))}
              </Box>

      </Box>
    );
  };
  
  export default Profile;