import { Box, Button, IconButton, Typography, useTheme } from "@mui/material";
import { tokens } from "../../theme";
import { mockTransactions } from "../../data/mockData";
import DownloadOutlinedIcon from "@mui/icons-material/DownloadOutlined";
import EmailIcon from "@mui/icons-material/Email";
import PointOfSaleIcon from "@mui/icons-material/PointOfSale";
import PersonAddIcon from "@mui/icons-material/PersonAdd";
import TrafficIcon from "@mui/icons-material/Traffic";
import Header from "../../components/Header";

import StatBox from "../../components/StatBox";
import ProgressCircle from "../../components/ProgressCircle";


const Contacts = () => {
    const theme = useTheme();
    const colors = tokens(theme.palette.mode);
  
    return (
      <Box m="20px">
        {/* HEADER */}
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Header title="CONTACT US" subtitle="" />
  
          {/* <Box>
            <Button
              sx={{
                backgroundColor: colors.blueAccent[700],
                color: colors.grey[100],
                fontSize: "14px",
                fontWeight: "bold",
                padding: "10px 20px",
              }}
            >
              <DownloadOutlinedIcon sx={{ mr: "10px" }} />
              Download Reports
            </Button>
          </Box> */}

          
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
            gridColumn="span 4"
            gridRow="span 2"
            backgroundColor={colors.primary[400]}
            p="30px"
          >
            
            <Box
              display="flex"
              flexDirection="column"
              alignItems="center"
              mt="25px"
            >
              <Box display="flex" justifyContent="center" alignItems="center">
                <img
                  alt="profile-user"
                  width="150px"
                  height="150px"
                  src={`../../assets/shreyansh.jpg`}
                  style={{ cursor: "pointer", borderRadius: "50%" }}
                />
              </Box>
              <Typography variant="h4" fontWeight="600" marginTop="3px">
                Shreyansh Aggarwal
                </Typography>
              {theme.palette.mode === "dark" ? (
                <Typography
                variant="h5"
                color={colors.greenAccent[500]}
                sx={{ mt: "15px" }}
              >
                shreyansh@gmail.com
              </Typography>
                ) : (
                <Typography
                    variant="h5"
                    color={colors.greenAccent[200]}
                    sx={{ mt: "15px" }}
                >
                    shreyansh@gmail.com
                </Typography>
                )}
        
              
              <Typography>+91 7840051946</Typography>
            </Box>
          </Box>

          <Box
            gridColumn="span 4"
            gridRow="span 2"
            backgroundColor={colors.primary[400]}
            p="30px"
          >
            

            <Box
              display="flex"
              flexDirection="column"
              alignItems="center"
              mt="25px"
            >
            {/* new id here */}
            <Box display="flex" justifyContent="center" alignItems="center">
                <img
                  alt="profile-user"
                  width="150px"
                  height="150px"
                  src={`../../assets/vishu.jpg`}
                  style={{ cursor: "pointer", borderRadius: "50%" }}
                />
              </Box>
              <Typography variant="h4" fontWeight="600" marginTop="3px">
                Vishu Aasliya
                </Typography>
              {theme.palette.mode === "dark" ? (
                <Typography
                variant="h5"
                color={colors.greenAccent[500]}
                sx={{ mt: "15px" }}
              >
                Vishu@gmail.com
              </Typography>
                ) : (
                <Typography
                    variant="h5"
                    color={colors.greenAccent[200]}
                    sx={{ mt: "15px" }}
                >
                    Vishu@gmail.com
                </Typography>
                )}
        
              
              <Typography>+91 7015665304</Typography>
            </Box>
          </Box>
          <Box
            gridColumn="span 4"
            gridRow="span 2"
            backgroundColor={colors.primary[400]}
            padding="30px"
          >
            
            
            <Box
              display="flex"
              flexDirection="column"
              alignItems="center"
              mt="25px"
            >
            <Box display="flex" justifyContent="center" alignItems="center">
                <img
                  alt="profile-user"
                  width="150px"
                  height="150px"
                  src={`../../assets/gautam.png`}
                  style={{ cursor: "pointer", borderRadius: "50%" }}
                />
              </Box>

              <Typography
                variant="h4"
                fontWeight="600"
                marginTop="3px"
                // sx={{ marginBottom: "15px" }}
                >
                Gautam Kanojiya
                </Typography>

              {theme.palette.mode === "dark" ? (
                <Typography
                variant="h5"
                color={colors.greenAccent[500]}
                sx={{ mt: "15px" }}
              >
                gkanojiya203@gmail.com
              </Typography>
                ) : (
                <Typography
                    variant="h5"
                    color={colors.greenAccent[200]}
                    sx={{ mt: "15px" }}
                >
                    gkanojiya203@gmail.com
                </Typography>
                )}
        
              
              <Typography>+91 7666424833</Typography>
              </Box>
          </Box>
        </Box>
      </Box>
    );
  };
  
  export default Contacts;