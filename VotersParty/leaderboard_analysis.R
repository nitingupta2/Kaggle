library(ggplot2)

leaderboard = read.csv('leaderboard.csv')

p = ggplot(leaderboard)

p + geom_density(aes(ScorePrivate, fill='Private LB'), alpha=0.2) +
	geom_density(aes(ScorePublic, fill='Public LB'), alpha=0.2) +
	geom_vline(aes(xintercept=median(ScorePrivate)), linetype='dashed', color='red') +
	geom_vline(aes(xintercept=median(ScorePublic)), linetype='dashed', color='blue') +
	scale_x_continuous(limits = c(0.55, 0.70), breaks = seq(0.55, 0.7, 0.01)) +
	ggtitle("Leaderboard Scores") +
	labs(x="Score") +
	theme(axis.title.y=element_blank(),
		  axis.text.y=element_blank(),
		  legend.title=element_blank())

median(leaderboard$ScorePrivate) - median(leaderboard$ScorePublic)

p + geom_point(aes(RankPublic, RankPrivate, color=abs(RankDelta) < 150)) +
	scale_color_manual(values=c('red', 'blue'), name='Public Rank\nDifference < +/- 150') +
	labs(x="Public Ranking", y="Private Ranking")
	
ggplot(leaderboard[1:300,]) + 
	geom_point(aes(RankPublic, RankPrivate, color= RankPublic > 300)) +
	scale_color_manual(values=c('blue', 'red'), name='Public Rank > 300') +
	labs(x="Public Ranking", y="Private Ranking")

nrow(leaderboard[leaderboard$RankPrivate <= 300 & leaderboard$RankPublic <= 300,])/300