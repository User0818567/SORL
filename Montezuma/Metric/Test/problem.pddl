(define (problem problem1) (:domain Montezuma)
(:objects 
        spot - MiddleLadder
        leftdoor - LeftDoor
        rightdoor - RightDoor
        keyloc - Key
        leftladder - LeftLadder
        rightladder - RightLadder
)
(:metric minimize( quality) )
(:goal (and
    ;todo: put the goal condition here
    (gameover)
))
(:init	(keyexist)
	(not(actorwithkey) )
	(at MiddleLadder)
	(= (quality) 0) )
)
