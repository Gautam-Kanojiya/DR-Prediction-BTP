import { Typography, Box, useTheme } from "@mui/material";
import { tokens } from "../theme";

const Header = ({ title, subtitle }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  
  return (
    <Box mb="30px">
      <Typography
        variant="h2"
        color={colors.grey[100]}
        fontWeight="bold"
        sx={{ m: "0 0 5px 10px" }}
      >
        {title}
      </Typography>
      

      {theme.palette.mode === "dark" ? (
              <Typography variant="h5" sx={{ color: colors.greenAccent[500] }} ml="10px">
              {subtitle}
            </Typography>
          ) : (
            <Typography variant="h5" sx={{ color: colors.greenAccent[200] }} ml="10px">
            {subtitle}
          </Typography>
          )}
        
      {/* <Typography variant="h5" color={colors.greenAccent[200]}>
        {subtitle}
      </Typography> */}
      
      
    </Box>
  );
};

export default Header;