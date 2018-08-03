
require(tidyverse)
require(tweedie)
require(ggridges)

df = tibble( p = seq(1,2,0.05)
             , rwn = row_number(p)
             , sin = sin(rwn) ) %>%
  mutate( data = map(p, function(p) rtweedie(500
                                             , mu = 1
                                             , phi = 1
                                             , power = p)  ) ) %>%
  unnest(data) 

p = df %>%
  filter( data <= 4) %>%
  mutate( data = ( 4 * abs( sin(rwn) ) ) - data ) %>%
  ggplot(aes(x = data, y = as.factor(p), fill = ..x.. ) ) +
    geom_density_ridges_gradient( color = 'white'
                                 , size = 0.5
                                 , scale = 3) +
    theme( panel.background = element_rect(fill = 'black')
           , panel.grid = element_blank()
           , aspect.ratio = 1
           , axis.title = element_blank()
           , axis.text = element_blank()
           , axis.ticks = element_blank()
           , legend.position = 'none') +
   xlim(-1,5) +
   scale_fill_viridis_c(option = "inferno") +
   scale_y_discrete( expand = c(0,5) )


ggsave(filename = './images/logo.png', dpi = 600)
