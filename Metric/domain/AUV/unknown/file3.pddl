(define (problem rover)
(:domain Rover)
(:objects 
	general - lander
	colour high_res low_res - mode
	rover0 - rover
	rover0store - store
	camera0 - camera
	 waypoint0 - waypoint
	 waypoint1 - waypoint
	 objective0 - objective
	x0 - loc_x
	x1 - loc_x
	x2 - loc_x
	x3 - loc_x
	x4 - loc_x
	x5 - loc_x
	x6 - loc_x
	x7 - loc_x
	x8 - loc_x
	x9 - loc_x
	x10 - loc_x
	x11 - loc_x
	x12 - loc_x
	x13 - loc_x
	x14 - loc_x
	x15 - loc_x
	y0 - loc_y
	y1 - loc_y
	y2 - loc_y
	y3 - loc_y
	y4 - loc_y
	y5 - loc_y
	y6 - loc_y
	y7 - loc_y
	y8 - loc_y
	y9 - loc_y
	y10 - loc_y
	y11 - loc_y
	y12 - loc_y
	y13 - loc_y
	y14 - loc_y
	y15 - loc_y)
(:init
	(= (total_time) 0)
	(= (distance) 0)
	(channel_free general)
	(store_of rover0store rover0)
	(empty rover0store)
	(on_board camera0 rover0)
	(supports camera0 colour)
	(supports camera0 high_res)
	(equipped_for_soil_analysis rover0)
	(equipped_for_rock_analysis rover0)
	(equipped_for_imaging rover0)
	(on_board camera0 rover0)
	(calibration_target camera0 objective0)
	(at_x x0)
	(at_y y0)
	(in_x x1 waypoint0)
	(in_x x2 waypoint0)
	(in_y y1 waypoint0)
	(in_y y2 waypoint0)
	(obstacle x1 y0)
	(obstacle x1 y1)
	(obstacle x2 y0)
	(obstacle x2 y1)
	(obstacle x6 y6)
	(obstacle x6 y7)
	(obstacle x7 y6)
	(obstacle x7 y7)
	(in_x x3 waypoint1)
	(in_x x4 waypoint1)
	(in_y y3 waypoint1)
	(in_y y4 waypoint1)
	(obstacle x1 y0)
	(obstacle x1 y1)
	(obstacle x2 y0)
	(obstacle x2 y1)
	(obstacle x6 y6)
	(obstacle x6 y7)
	(obstacle x7 y6)
	(obstacle x7 y7)
	(link_x x2 x0)
	(link_x x3 x1)
	(link_x x0 x2)
	(link_x x4 x2)
	(link_x x1 x3)
	(link_x x5 x3)
	(link_x x2 x4)
	(link_x x6 x4)
	(link_x x3 x5)
	(link_x x7 x5)
	(link_x x4 x6)
	(link_x x8 x6)
	(link_x x5 x7)
	(link_x x9 x7)
	(link_x x6 x8)
	(link_x x10 x8)
	(link_x x7 x9)
	(link_x x11 x9)
	(link_x x8 x10)
	(link_x x12 x10)
	(link_x x9 x11)
	(link_x x13 x11)
	(link_x x10 x12)
	(link_x x14 x12)
	(link_x x11 x13)
	(link_x x12 x14)
	(link_y y2 y0)
	(link_y y3 y1)
	(link_y y0 y2)
	(link_y y4 y2)
	(link_y y1 y3)
	(link_y y5 y3)
	(link_y y2 y4)
	(link_y y6 y4)
	(link_y y3 y5)
	(link_y y7 y5)
	(link_y y4 y6)
	(link_y y8 y6)
	(link_y y5 y7)
	(link_y y9 y7)
	(link_y y6 y8)
	(link_y y10 y8)
	(link_y y7 y9)
	(link_y y11 y9)
	(link_y y8 y10)
	(link_y y12 y10)
	(link_y y9 y11)
	(link_y y13 y11)
	(link_y y10 y12)
	(link_y y14 y12)
	(link_y y11 y13)
	(link_y y12 y14)
	(diagonallink x0 x2 y1 y3)
	(diagonallink x0 x2 y2 y4)
	(diagonallink x0 x2 y3 y5)
	(diagonallink x0 x2 y4 y6)
	(diagonallink x0 x2 y4 y2)
	(diagonallink x0 x2 y5 y7)
	(diagonallink x0 x2 y5 y3)
	(diagonallink x0 x2 y6 y4)
	(diagonallink x0 x2 y6 y8)
	(diagonallink x0 x2 y7 y5)
	(diagonallink x0 x2 y7 y9)
	(diagonallink x0 x2 y8 y10)
	(diagonallink x0 x2 y8 y6)
	(diagonallink x0 x2 y9 y11)
	(diagonallink x0 x2 y9 y7)
	(diagonallink x0 x2 y10 y12)
	(diagonallink x0 x2 y10 y8)
	(diagonallink x0 x2 y11 y9)
	(diagonallink x0 x2 y11 y13)
	(diagonallink x0 x2 y12 y14)
	(diagonallink x0 x2 y12 y10)
	(diagonallink x0 x2 y13 y11)
	(diagonallink x0 x2 y14 y12)
	(diagonallink x1 x3 y2 y4)
	(diagonallink x1 x3 y3 y5)
	(diagonallink x1 x3 y3 y1)
	(diagonallink x1 x3 y4 y6)
	(diagonallink x1 x3 y4 y2)
	(diagonallink x1 x3 y5 y7)
	(diagonallink x1 x3 y5 y3)
	(diagonallink x1 x3 y6 y8)
	(diagonallink x1 x3 y6 y4)
	(diagonallink x1 x3 y7 y9)
	(diagonallink x1 x3 y7 y5)
	(diagonallink x1 x3 y8 y6)
	(diagonallink x1 x3 y8 y10)
	(diagonallink x1 x3 y9 y7)
	(diagonallink x1 x3 y9 y11)
	(diagonallink x1 x3 y10 y8)
	(diagonallink x1 x3 y10 y12)
	(diagonallink x1 x3 y11 y13)
	(diagonallink x1 x3 y11 y9)
	(diagonallink x1 x3 y12 y14)
	(diagonallink x1 x3 y12 y10)
	(diagonallink x1 x3 y13 y11)
	(diagonallink x1 x3 y14 y12)
	(diagonallink x2 x4 y2 y4)
	(diagonallink x2 x4 y3 y5)
	(diagonallink x2 x4 y3 y1)
	(diagonallink x2 x4 y4 y6)
	(diagonallink x2 x4 y4 y2)
	(diagonallink x2 x4 y5 y3)
	(diagonallink x2 x4 y5 y7)
	(diagonallink x2 x4 y6 y4)
	(diagonallink x2 x4 y6 y8)
	(diagonallink x2 x4 y7 y9)
	(diagonallink x2 x4 y7 y5)
	(diagonallink x2 x4 y8 y10)
	(diagonallink x2 x4 y8 y6)
	(diagonallink x2 x4 y9 y7)
	(diagonallink x2 x4 y9 y11)
	(diagonallink x2 x4 y10 y12)
	(diagonallink x2 x4 y10 y8)
	(diagonallink x2 x4 y11 y9)
	(diagonallink x2 x4 y11 y13)
	(diagonallink x2 x4 y12 y14)
	(diagonallink x2 x4 y12 y10)
	(diagonallink x2 x4 y13 y11)
	(diagonallink x2 x4 y14 y12)
	(diagonallink x3 x5 y0 y2)
	(diagonallink x3 x5 y1 y3)
	(diagonallink x3 x1 y1 y3)
	(diagonallink x3 x1 y2 y4)
	(diagonallink x3 x5 y2 y4)
	(diagonallink x3 x1 y3 y5)
	(diagonallink x3 x5 y3 y1)
	(diagonallink x3 x5 y3 y5)
	(diagonallink x3 x1 y4 y2)
	(diagonallink x3 x1 y4 y6)
	(diagonallink x3 x5 y4 y6)
	(diagonallink x3 x5 y4 y2)
	(diagonallink x3 x5 y5 y7)
	(diagonallink x3 x1 y5 y7)
	(diagonallink x3 x1 y5 y3)
	(diagonallink x3 x5 y5 y3)
	(diagonallink x3 x1 y6 y4)
	(diagonallink x3 x5 y6 y8)
	(diagonallink x3 x5 y6 y4)
	(diagonallink x3 x1 y6 y8)
	(diagonallink x3 x1 y7 y5)
	(diagonallink x3 x5 y7 y5)
	(diagonallink x3 x1 y7 y9)
	(diagonallink x3 x5 y7 y9)
	(diagonallink x3 x1 y8 y6)
	(diagonallink x3 x5 y8 y10)
	(diagonallink x3 x5 y8 y6)
	(diagonallink x3 x1 y8 y10)
	(diagonallink x3 x5 y9 y11)
	(diagonallink x3 x5 y9 y7)
	(diagonallink x3 x1 y9 y11)
	(diagonallink x3 x1 y9 y7)
	(diagonallink x3 x1 y10 y12)
	(diagonallink x3 x5 y10 y8)
	(diagonallink x3 x1 y10 y8)
	(diagonallink x3 x5 y10 y12)
	(diagonallink x3 x5 y11 y13)
	(diagonallink x3 x5 y11 y9)
	(diagonallink x3 x1 y11 y9)
	(diagonallink x3 x1 y11 y13)
	(diagonallink x3 x5 y12 y14)
	(diagonallink x3 x1 y12 y14)
	(diagonallink x3 x5 y12 y10)
	(diagonallink x3 x1 y12 y10)
	(diagonallink x3 x5 y13 y11)
	(diagonallink x3 x1 y13 y11)
	(diagonallink x3 x1 y14 y12)
	(diagonallink x3 x5 y14 y12)
	(diagonallink x4 x6 y0 y2)
	(diagonallink x4 x2 y0 y2)
	(diagonallink x4 x6 y1 y3)
	(diagonallink x4 x2 y1 y3)
	(diagonallink x4 x2 y2 y4)
	(diagonallink x4 x6 y2 y4)
	(diagonallink x4 x2 y3 y5)
	(diagonallink x4 x6 y3 y5)
	(diagonallink x4 x6 y3 y1)
	(diagonallink x4 x2 y4 y6)
	(diagonallink x4 x6 y4 y2)
	(diagonallink x4 x2 y4 y2)
	(diagonallink x4 x6 y5 y3)
	(diagonallink x4 x2 y5 y3)
	(diagonallink x4 x2 y5 y7)
	(diagonallink x4 x2 y6 y8)
	(diagonallink x4 x6 y6 y4)
	(diagonallink x4 x2 y6 y4)
	(diagonallink x4 x6 y6 y8)
	(diagonallink x4 x2 y7 y9)
	(diagonallink x4 x6 y7 y5)
	(diagonallink x4 x2 y7 y5)
	(diagonallink x4 x6 y7 y9)
	(diagonallink x4 x2 y8 y10)
	(diagonallink x4 x6 y8 y10)
	(diagonallink x4 x2 y8 y6)
	(diagonallink x4 x6 y9 y11)
	(diagonallink x4 x2 y9 y11)
	(diagonallink x4 x2 y9 y7)
	(diagonallink x4 x6 y10 y12)
	(diagonallink x4 x6 y10 y8)
	(diagonallink x4 x2 y10 y12)
	(diagonallink x4 x2 y10 y8)
	(diagonallink x4 x2 y11 y13)
	(diagonallink x4 x6 y11 y13)
	(diagonallink x4 x6 y11 y9)
	(diagonallink x4 x2 y11 y9)
	(diagonallink x4 x2 y12 y14)
	(diagonallink x4 x2 y12 y10)
	(diagonallink x4 x6 y12 y10)
	(diagonallink x4 x6 y12 y14)
	(diagonallink x4 x2 y13 y11)
	(diagonallink x4 x6 y13 y11)
	(diagonallink x4 x6 y14 y12)
	(diagonallink x4 x2 y14 y12)
	(diagonallink x5 x3 y0 y2)
	(diagonallink x5 x7 y0 y2)
	(diagonallink x5 x7 y1 y3)
	(diagonallink x5 x3 y1 y3)
	(diagonallink x5 x7 y2 y4)
	(diagonallink x5 x3 y2 y4)
	(diagonallink x5 x7 y3 y5)
	(diagonallink x5 x3 y3 y5)
	(diagonallink x5 x3 y3 y1)
	(diagonallink x5 x7 y3 y1)
	(diagonallink x5 x3 y4 y6)
	(diagonallink x5 x7 y4 y2)
	(diagonallink x5 x3 y4 y2)
	(diagonallink x5 x3 y5 y3)
	(diagonallink x5 x3 y5 y7)
	(diagonallink x5 x7 y5 y3)
	(diagonallink x5 x3 y6 y4)
	(diagonallink x5 x3 y6 y8)
	(diagonallink x5 x7 y6 y4)
	(diagonallink x5 x7 y7 y9)
	(diagonallink x5 x3 y7 y5)
	(diagonallink x5 x3 y7 y9)
	(diagonallink x5 x3 y8 y10)
	(diagonallink x5 x3 y8 y6)
	(diagonallink x5 x7 y8 y10)
	(diagonallink x5 x3 y9 y11)
	(diagonallink x5 x3 y9 y7)
	(diagonallink x5 x7 y9 y11)
	(diagonallink x5 x7 y10 y8)
	(diagonallink x5 x3 y10 y8)
	(diagonallink x5 x3 y10 y12)
	(diagonallink x5 x7 y10 y12)
	(diagonallink x5 x3 y11 y9)
	(diagonallink x5 x7 y11 y13)
	(diagonallink x5 x3 y11 y13)
	(diagonallink x5 x7 y11 y9)
	(diagonallink x5 x7 y12 y14)
	(diagonallink x5 x3 y12 y14)
	(diagonallink x5 x3 y12 y10)
	(diagonallink x5 x7 y12 y10)
	(diagonallink x5 x7 y13 y11)
	(diagonallink x5 x3 y13 y11)
	(diagonallink x5 x7 y14 y12)
	(diagonallink x5 x3 y14 y12)
	(diagonallink x6 x8 y0 y2)
	(diagonallink x6 x4 y0 y2)
	(diagonallink x6 x8 y1 y3)
	(diagonallink x6 x4 y1 y3)
	(diagonallink x6 x8 y2 y4)
	(diagonallink x6 x4 y2 y4)
	(diagonallink x6 x8 y3 y1)
	(diagonallink x6 x4 y3 y1)
	(diagonallink x6 x8 y3 y5)
	(diagonallink x6 x4 y3 y5)
	(diagonallink x6 x4 y4 y2)
	(diagonallink x6 x4 y4 y6)
	(diagonallink x6 x8 y4 y2)
	(diagonallink x6 x8 y4 y6)
	(diagonallink x6 x8 y5 y3)
	(diagonallink x6 x4 y5 y3)
	(diagonallink x6 x4 y5 y7)
	(diagonallink x6 x4 y8 y6)
	(diagonallink x6 x4 y8 y10)
	(diagonallink x6 x8 y8 y10)
	(diagonallink x6 x8 y9 y7)
	(diagonallink x6 x4 y9 y7)
	(diagonallink x6 x4 y9 y11)
	(diagonallink x6 x8 y9 y11)
	(diagonallink x6 x4 y10 y8)
	(diagonallink x6 x8 y10 y8)
	(diagonallink x6 x8 y10 y12)
	(diagonallink x6 x4 y10 y12)
	(diagonallink x6 x8 y11 y13)
	(diagonallink x6 x8 y11 y9)
	(diagonallink x6 x4 y11 y9)
	(diagonallink x6 x4 y11 y13)
	(diagonallink x6 x4 y12 y14)
	(diagonallink x6 x4 y12 y10)
	(diagonallink x6 x8 y12 y10)
	(diagonallink x6 x8 y12 y14)
	(diagonallink x6 x8 y13 y11)
	(diagonallink x6 x4 y13 y11)
	(diagonallink x6 x4 y14 y12)
	(diagonallink x6 x8 y14 y12)
	(diagonallink x7 x9 y0 y2)
	(diagonallink x7 x5 y0 y2)
	(diagonallink x7 x9 y1 y3)
	(diagonallink x7 x5 y1 y3)
	(diagonallink x7 x9 y2 y4)
	(diagonallink x7 x5 y2 y4)
	(diagonallink x7 x9 y3 y5)
	(diagonallink x7 x9 y3 y1)
	(diagonallink x7 x5 y3 y1)
	(diagonallink x7 x5 y3 y5)
	(diagonallink x7 x9 y4 y6)
	(diagonallink x7 x9 y4 y2)
	(diagonallink x7 x5 y4 y6)
	(diagonallink x7 x5 y4 y2)
	(diagonallink x7 x9 y5 y7)
	(diagonallink x7 x9 y5 y3)
	(diagonallink x7 x5 y5 y3)
	(diagonallink x7 x5 y8 y10)
	(diagonallink x7 x9 y8 y10)
	(diagonallink x7 x9 y8 y6)
	(diagonallink x7 x5 y9 y7)
	(diagonallink x7 x9 y9 y11)
	(diagonallink x7 x5 y9 y11)
	(diagonallink x7 x9 y9 y7)
	(diagonallink x7 x9 y10 y12)
	(diagonallink x7 x5 y10 y8)
	(diagonallink x7 x9 y10 y8)
	(diagonallink x7 x5 y10 y12)
	(diagonallink x7 x5 y11 y9)
	(diagonallink x7 x9 y11 y9)
	(diagonallink x7 x5 y11 y13)
	(diagonallink x7 x9 y11 y13)
	(diagonallink x7 x9 y12 y14)
	(diagonallink x7 x9 y12 y10)
	(diagonallink x7 x5 y12 y14)
	(diagonallink x7 x5 y12 y10)
	(diagonallink x7 x5 y13 y11)
	(diagonallink x7 x9 y13 y11)
	(diagonallink x7 x5 y14 y12)
	(diagonallink x7 x9 y14 y12)
	(diagonallink x8 x10 y0 y2)
	(diagonallink x8 x6 y0 y2)
	(diagonallink x8 x10 y1 y3)
	(diagonallink x8 x6 y1 y3)
	(diagonallink x8 x10 y2 y4)
	(diagonallink x8 x6 y2 y4)
	(diagonallink x8 x10 y3 y5)
	(diagonallink x8 x6 y3 y1)
	(diagonallink x8 x10 y3 y1)
	(diagonallink x8 x6 y3 y5)
	(diagonallink x8 x6 y4 y2)
	(diagonallink x8 x10 y4 y6)
	(diagonallink x8 x10 y4 y2)
	(diagonallink x8 x6 y5 y3)
	(diagonallink x8 x10 y5 y3)
	(diagonallink x8 x10 y5 y7)
	(diagonallink x8 x6 y6 y4)
	(diagonallink x8 x10 y6 y4)
	(diagonallink x8 x10 y6 y8)
	(diagonallink x8 x10 y7 y5)
	(diagonallink x8 x6 y7 y9)
	(diagonallink x8 x10 y7 y9)
	(diagonallink x8 x10 y8 y10)
	(diagonallink x8 x6 y8 y10)
	(diagonallink x8 x10 y8 y6)
	(diagonallink x8 x10 y9 y7)
	(diagonallink x8 x6 y9 y11)
	(diagonallink x8 x10 y9 y11)
	(diagonallink x8 x6 y10 y12)
	(diagonallink x8 x6 y10 y8)
	(diagonallink x8 x10 y10 y8)
	(diagonallink x8 x10 y10 y12)
	(diagonallink x8 x10 y11 y13)
	(diagonallink x8 x6 y11 y9)
	(diagonallink x8 x10 y11 y9)
	(diagonallink x8 x6 y11 y13)
	(diagonallink x8 x10 y12 y10)
	(diagonallink x8 x6 y12 y10)
	(diagonallink x8 x6 y12 y14)
	(diagonallink x8 x10 y12 y14)
	(diagonallink x8 x10 y13 y11)
	(diagonallink x8 x6 y13 y11)
	(diagonallink x8 x10 y14 y12)
	(diagonallink x8 x6 y14 y12)
	(diagonallink x9 x11 y0 y2)
	(diagonallink x9 x7 y0 y2)
	(diagonallink x9 x11 y1 y3)
	(diagonallink x9 x7 y1 y3)
	(diagonallink x9 x7 y2 y4)
	(diagonallink x9 x11 y2 y4)
	(diagonallink x9 x7 y3 y1)
	(diagonallink x9 x11 y3 y5)
	(diagonallink x9 x11 y3 y1)
	(diagonallink x9 x7 y3 y5)
	(diagonallink x9 x11 y4 y6)
	(diagonallink x9 x7 y4 y2)
	(diagonallink x9 x11 y4 y2)
	(diagonallink x9 x11 y5 y7)
	(diagonallink x9 x11 y5 y3)
	(diagonallink x9 x7 y5 y3)
	(diagonallink x9 x7 y6 y4)
	(diagonallink x9 x7 y6 y8)
	(diagonallink x9 x11 y6 y4)
	(diagonallink x9 x11 y6 y8)
	(diagonallink x9 x11 y7 y5)
	(diagonallink x9 x7 y7 y9)
	(diagonallink x9 x11 y7 y9)
	(diagonallink x9 x7 y7 y5)
	(diagonallink x9 x11 y8 y10)
	(diagonallink x9 x11 y8 y6)
	(diagonallink x9 x7 y8 y10)
	(diagonallink x9 x11 y9 y11)
	(diagonallink x9 x11 y9 y7)
	(diagonallink x9 x7 y9 y11)
	(diagonallink x9 x7 y10 y12)
	(diagonallink x9 x11 y10 y12)
	(diagonallink x9 x11 y10 y8)
	(diagonallink x9 x7 y10 y8)
	(diagonallink x9 x11 y11 y13)
	(diagonallink x9 x7 y11 y9)
	(diagonallink x9 x11 y11 y9)
	(diagonallink x9 x7 y11 y13)
	(diagonallink x9 x7 y12 y10)
	(diagonallink x9 x11 y12 y10)
	(diagonallink x9 x7 y12 y14)
	(diagonallink x9 x11 y12 y14)
	(diagonallink x9 x7 y13 y11)
	(diagonallink x9 x11 y13 y11)
	(diagonallink x9 x11 y14 y12)
	(diagonallink x9 x7 y14 y12)
	(diagonallink x10 x12 y0 y2)
	(diagonallink x10 x8 y0 y2)
	(diagonallink x10 x12 y1 y3)
	(diagonallink x10 x8 y1 y3)
	(diagonallink x10 x12 y2 y4)
	(diagonallink x10 x8 y2 y4)
	(diagonallink x10 x12 y3 y1)
	(diagonallink x10 x8 y3 y5)
	(diagonallink x10 x8 y3 y1)
	(diagonallink x10 x12 y3 y5)
	(diagonallink x10 x12 y4 y6)
	(diagonallink x10 x12 y4 y2)
	(diagonallink x10 x8 y4 y2)
	(diagonallink x10 x8 y4 y6)
	(diagonallink x10 x8 y5 y7)
	(diagonallink x10 x8 y5 y3)
	(diagonallink x10 x12 y5 y3)
	(diagonallink x10 x12 y5 y7)
	(diagonallink x10 x8 y6 y8)
	(diagonallink x10 x8 y6 y4)
	(diagonallink x10 x12 y6 y8)
	(diagonallink x10 x12 y6 y4)
	(diagonallink x10 x12 y7 y9)
	(diagonallink x10 x12 y7 y5)
	(diagonallink x10 x8 y7 y9)
	(diagonallink x10 x8 y7 y5)
	(diagonallink x10 x8 y8 y6)
	(diagonallink x10 x8 y8 y10)
	(diagonallink x10 x12 y8 y10)
	(diagonallink x10 x12 y8 y6)
	(diagonallink x10 x12 y9 y7)
	(diagonallink x10 x8 y9 y11)
	(diagonallink x10 x8 y9 y7)
	(diagonallink x10 x12 y9 y11)
	(diagonallink x10 x12 y10 y8)
	(diagonallink x10 x8 y10 y8)
	(diagonallink x10 x8 y10 y12)
	(diagonallink x10 x12 y10 y12)
	(diagonallink x10 x12 y11 y9)
	(diagonallink x10 x12 y11 y13)
	(diagonallink x10 x8 y11 y13)
	(diagonallink x10 x8 y11 y9)
	(diagonallink x10 x12 y12 y14)
	(diagonallink x10 x8 y12 y14)
	(diagonallink x10 x12 y12 y10)
	(diagonallink x10 x8 y12 y10)
	(diagonallink x10 x12 y13 y11)
	(diagonallink x10 x8 y13 y11)
	(diagonallink x10 x8 y14 y12)
	(diagonallink x10 x12 y14 y12)
	(diagonallink x11 x9 y0 y2)
	(diagonallink x11 x13 y0 y2)
	(diagonallink x11 x13 y1 y3)
	(diagonallink x11 x9 y1 y3)
	(diagonallink x11 x9 y2 y4)
	(diagonallink x11 x13 y2 y4)
	(diagonallink x11 x13 y3 y1)
	(diagonallink x11 x13 y3 y5)
	(diagonallink x11 x9 y3 y1)
	(diagonallink x11 x9 y3 y5)
	(diagonallink x11 x9 y4 y2)
	(diagonallink x11 x9 y4 y6)
	(diagonallink x11 x13 y4 y6)
	(diagonallink x11 x13 y4 y2)
	(diagonallink x11 x9 y5 y7)
	(diagonallink x11 x13 y5 y7)
	(diagonallink x11 x13 y5 y3)
	(diagonallink x11 x9 y5 y3)
	(diagonallink x11 x13 y6 y4)
	(diagonallink x11 x13 y6 y8)
	(diagonallink x11 x9 y6 y4)
	(diagonallink x11 x9 y6 y8)
	(diagonallink x11 x13 y7 y5)
	(diagonallink x11 x9 y7 y9)
	(diagonallink x11 x9 y7 y5)
	(diagonallink x11 x13 y7 y9)
	(diagonallink x11 x9 y8 y6)
	(diagonallink x11 x9 y8 y10)
	(diagonallink x11 x13 y8 y10)
	(diagonallink x11 x13 y8 y6)
	(diagonallink x11 x13 y9 y11)
	(diagonallink x11 x9 y9 y7)
	(diagonallink x11 x13 y9 y7)
	(diagonallink x11 x9 y9 y11)
	(diagonallink x11 x13 y10 y8)
	(diagonallink x11 x9 y10 y12)
	(diagonallink x11 x13 y10 y12)
	(diagonallink x11 x9 y10 y8)
	(diagonallink x11 x13 y11 y9)
	(diagonallink x11 x9 y11 y13)
	(diagonallink x11 x9 y11 y9)
	(diagonallink x11 x13 y11 y13)
	(diagonallink x11 x13 y12 y10)
	(diagonallink x11 x13 y12 y14)
	(diagonallink x11 x9 y12 y14)
	(diagonallink x11 x9 y12 y10)
	(diagonallink x11 x13 y13 y11)
	(diagonallink x11 x9 y13 y11)
	(diagonallink x11 x13 y14 y12)
	(diagonallink x11 x9 y14 y12)
	(diagonallink x12 x14 y0 y2)
	(diagonallink x12 x10 y0 y2)
	(diagonallink x12 x14 y1 y3)
	(diagonallink x12 x10 y1 y3)
	(diagonallink x12 x14 y2 y4)
	(diagonallink x12 x10 y2 y4)
	(diagonallink x12 x14 y3 y1)
	(diagonallink x12 x10 y3 y1)
	(diagonallink x12 x14 y3 y5)
	(diagonallink x12 x10 y3 y5)
	(diagonallink x12 x10 y4 y6)
	(diagonallink x12 x14 y4 y2)
	(diagonallink x12 x10 y4 y2)
	(diagonallink x12 x14 y4 y6)
	(diagonallink x12 x10 y5 y7)
	(diagonallink x12 x14 y5 y7)
	(diagonallink x12 x14 y5 y3)
	(diagonallink x12 x10 y5 y3)
	(diagonallink x12 x10 y6 y4)
	(diagonallink x12 x10 y6 y8)
	(diagonallink x12 x14 y6 y8)
	(diagonallink x12 x14 y6 y4)
	(diagonallink x12 x14 y7 y5)
	(diagonallink x12 x10 y7 y9)
	(diagonallink x12 x10 y7 y5)
	(diagonallink x12 x14 y7 y9)
	(diagonallink x12 x10 y8 y6)
	(diagonallink x12 x10 y8 y10)
	(diagonallink x12 x14 y8 y10)
	(diagonallink x12 x14 y8 y6)
	(diagonallink x12 x10 y9 y7)
	(diagonallink x12 x14 y9 y7)
	(diagonallink x12 x14 y9 y11)
	(diagonallink x12 x10 y9 y11)
	(diagonallink x12 x10 y10 y8)
	(diagonallink x12 x14 y10 y8)
	(diagonallink x12 x14 y10 y12)
	(diagonallink x12 x10 y10 y12)
	(diagonallink x12 x10 y11 y9)
	(diagonallink x12 x10 y11 y13)
	(diagonallink x12 x14 y11 y13)
	(diagonallink x12 x14 y11 y9)
	(diagonallink x12 x10 y12 y10)
	(diagonallink x12 x14 y12 y10)
	(diagonallink x12 x10 y12 y14)
	(diagonallink x12 x14 y12 y14)
	(diagonallink x12 x10 y13 y11)
	(diagonallink x12 x14 y13 y11)
	(diagonallink x12 x14 y14 y12)
	(diagonallink x12 x10 y14 y12)
	(diagonallink x13 x11 y0 y2)
	(diagonallink x13 x11 y1 y3)
	(diagonallink x13 x11 y2 y4)
	(diagonallink x13 x11 y3 y5)
	(diagonallink x13 x11 y3 y1)
	(diagonallink x13 x11 y4 y2)
	(diagonallink x13 x11 y4 y6)
	(diagonallink x13 x11 y5 y3)
	(diagonallink x13 x11 y5 y7)
	(diagonallink x13 x11 y6 y8)
	(diagonallink x13 x11 y6 y4)
	(diagonallink x13 x11 y7 y9)
	(diagonallink x13 x11 y7 y5)
	(diagonallink x13 x11 y8 y6)
	(diagonallink x13 x11 y8 y10)
	(diagonallink x13 x11 y9 y11)
	(diagonallink x13 x11 y9 y7)
	(diagonallink x13 x11 y10 y8)
	(diagonallink x13 x11 y10 y12)
	(diagonallink x13 x11 y11 y13)
	(diagonallink x13 x11 y11 y9)
	(diagonallink x13 x11 y12 y10)
	(diagonallink x13 x11 y12 y14)
	(diagonallink x13 x11 y13 y11)
	(diagonallink x13 x11 y14 y12)
	(diagonallink x14 x12 y0 y2)
	(diagonallink x14 x12 y1 y3)
	(diagonallink x14 x12 y2 y4)
	(diagonallink x14 x12 y3 y1)
	(diagonallink x14 x12 y3 y5)
	(diagonallink x14 x12 y4 y2)
	(diagonallink x14 x12 y4 y6)
	(diagonallink x14 x12 y5 y7)
	(diagonallink x14 x12 y5 y3)
	(diagonallink x14 x12 y6 y8)
	(diagonallink x14 x12 y6 y4)
	(diagonallink x14 x12 y7 y9)
	(diagonallink x14 x12 y7 y5)
	(diagonallink x14 x12 y8 y10)
	(diagonallink x14 x12 y8 y6)
	(diagonallink x14 x12 y9 y11)
	(diagonallink x14 x12 y9 y7)
	(diagonallink x14 x12 y10 y12)
	(diagonallink x14 x12 y10 y8)
	(diagonallink x14 x12 y11 y13)
	(diagonallink x14 x12 y11 y9)
	(diagonallink x14 x12 y12 y14)
	(diagonallink x14 x12 y12 y10)
	(diagonallink x14 x12 y13 y11)
	(diagonallink x14 x12 y14 y12)
	(at_soil_sample waypoint0)
	(at_rock_sample waypoint0)
	(at_lander general waypoint0)
	(at_lander general waypoint1)
	(visible_from objective0 waypoint1)
	(visible waypoint1 waypoint1)
	(visible waypoint0 waypoint1)
	(visible waypoint0 waypoint0)
	(visible waypoint1 waypoint0)
)
(:goal
(and
	(communicated_image_data objective0 high_res)
	(communicated_soil_data waypoint0)
	(communicated_rock_data waypoint0)
))
(:metric minimize(distance))
)
