(define (problem problem1) (:domain Montezuma)
(:objects 
        key - key
        spot - spot
        leftdoor - leftdoor
        rightdoor - rightdoor
        keyloc - keyloc
        leftladder - leftladder
        rightladder - rightladder
)

(:init
    (at spot)
    (keyexist)
    (not (actorwithkey))
    (not (gameover))
    (= (quality) 0)
    (= (lastquality) 3)
)

(:goal (and
    ;todo: put the goal condition here
    (gameover)
))

;un-comment the following line if metric is needed
(:metric minimize( quality) )
)
