
require(tidyverse)

ggplot(NULL)+
  theme( panel.background = element_rect(fill = '#77773c')
         , panel.grid = element_blank()
         , aspect.ratio = 1
         , axis.title = element_blank()
         , axis.text = element_blank()
         , axis.ticks = element_blank()
         , legend.position = 'none')

ggsave(filename = './themes/hugo-tranquilpeak-theme/static/images/cover.jpg', dpi = 600, width = 4, height = 4)

