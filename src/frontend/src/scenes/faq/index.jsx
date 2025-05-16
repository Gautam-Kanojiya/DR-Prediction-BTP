import { Box, useTheme } from "@mui/material";
import Header from "../../components/Header";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import Typography from "@mui/material/Typography";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { tokens } from "../../theme";

const FAQ = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  return (
    <Box m="20px">
      <Header title="FAQ" subtitle="Frequently Asked Questions Page" />
        
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
           How to manage your diabetes?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
          Make healthy eating and physical activity part of your daily routine. 
          Try to get at least 150 minutes of moderate aerobic activity, such as walking, each week. 
          Take oral diabetes medications or insulin as directed.
          </Typography>
        </AccordionDetails>
      </Accordion>
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
          What is Diabetic retinopathy?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
          Diabetic retinopathy is a diabetes complication that affects eyes. 
          It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye(retina).
          </Typography>
        </AccordionDetails>
      </Accordion>
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
          When to see an eye doctor?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
          If you have diabetes, see your eye doctor for a yearly eye exam with dilation — even if your vision seems fine.
          Developing diabetes when pregnant (gestational diabetes) or having diabetes before becoming pregnant can increase your risk of diabetic retinopathy. 
          If you're pregnant, your eye doctor might recommend additional eye exams throughout your pregnancy.
          Contact your eye doctor right away if your vision changes suddenly or becomes blurry, spotty or hazy.
          </Typography>
        </AccordionDetails>
      </Accordion>
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
          What are the types of diabetic retinopathy?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
          Early diabetic retinopathy and Advanced diabetic retinopathy
          </Typography>
        </AccordionDetails>
      </Accordion>
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
          What is Advanced diabetic retinopathy ?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
          In this type, damaged blood vessels close off, causing the growth of new, abnormal blood vessels in the retina. 
          These new blood vessels are fragile and can leak into the clear, jellylike substance that fills the center of your eye (vitreous).
          Eventually, scar tissue from the growth of new blood vessels can cause the retina to detach from the back of your eye. 
          If the new blood vessels interfere with the normal flow of fluid out of the eye, pressure can build in the eyeball. 
          This buildup can damage the nerve that carries images from your eye to your brain (optic nerve), resulting in glaucoma.
          </Typography>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default FAQ;