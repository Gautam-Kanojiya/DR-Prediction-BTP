import { Box, IconButton, Typography, useTheme } from "@mui/material";
import { useContext } from "react";
import { ColorModeContext, tokens } from "../../theme";
import Header from "../../components/Header";

import InputBase from "@mui/material/InputBase";
import LightModeOutlinedIcon from "@mui/icons-material/LightModeOutlined";
import DarkModeOutlinedIcon from "@mui/icons-material/DarkModeOutlined";
import NotificationsOutlinedIcon from "@mui/icons-material/NotificationsOutlined";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import PersonOutlinedIcon from "@mui/icons-material/PersonOutlined";
import SearchIcon from "@mui/icons-material/Search";
import { HomeOutlined } from "@mui/icons-material";
import { Link } from 'react-router-dom';


const Topbar = () => {
    const theme = useTheme();
    const colors = tokens(theme.palette.mode);
    const colorMode = useContext(ColorModeContext);


    return <Box display="flex" justifyContent="space-between" p={2} backgroundColor={colors.primary[400]}>
        {/* SEARCH BAR */}
      <Box
        display="flex"
        //backgroundColor={colors.primary[400]}
        borderRadius="3px"
        height="40px"
      >
        {/* <InputBase sx={{ ml: 2, flex: 1 }} placeholder="Search" />
        <IconButton type="button" sx={{ p: 1 }}>
          <SearchIcon />
        </IconButton> */}
        <img
                  alt="svg-logo"
                  // width="150px"
                  // height="150px"
                  src={`../../assets/logo_svg.png`}
                  style={{ cursor: "pointer", borderRadius: "50%" }}
                />
      </Box>

      <Box
        display="flex"
        //backgroundColor={colors.primary[400]}
        borderRadius="3px"
        height="40px"
      >
       <Header title="DIABETIC RETINOPATHY" subtitle="" />
      </Box>


        {/* ICONS */}
        <Box display="flex" mt="-10px">
            <IconButton onClick={colorMode.toggleColorMode}>
            {theme.palette.mode === "dark" ? (
                <DarkModeOutlinedIcon />
            ) : (
                <LightModeOutlinedIcon />
            )}
            </IconButton>
            
            <IconButton 
              component={Link} 
              to="https://healthplus.flipkart.com/"
              target="_blank"
              rel="noopener noreferrer"
            >

            <img
                  alt="profile-user"
                  width="45px"
                  height="45px"
                  src={`../../assets/flipkart.png`}
                  style={{ cursor: "pointer", borderRadius: "25%" }}
                />
            </IconButton>

            {/* <IconButton component={Link} to="/"> 
            <NotificationsOutlinedIcon />
            <HomeOutlined/>
            </IconButton> */}
            
            {/* <IconButton>
            <SettingsOutlinedIcon />
            </IconButton> */}
            <IconButton component={Link} to="/profile">
            <PersonOutlinedIcon />
            </IconButton>
        </Box>
    </Box>
}

export default Topbar;