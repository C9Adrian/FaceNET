import curses
#-----------------
# Curses Variables
#-----------------
stdscr = curses.initscr()  # Initiate the curses terminal
curses.start_color()
curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
curses.init_pair(3, curses.COLOR_BLUE, curses.COLOR_BLACK)
curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

k = 0
optNum = 1
name = ""

while True:
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    XCursor = width // 6
    YCursor = height // 6

    # Print title
    stdscr.attron(curses.color_pair(3))
    stdscr.attron(curses.A_BOLD)
    stdscr.addstr(YCursor, XCursor, "Photo Booth")
    stdscr.attroff(curses.color_pair(3))
    stdscr.attroff(curses.A_BOLD)

    # Print OPTIONS
    YCursor = YCursor + 2
    stdscr.attron(curses.A_ITALIC)
    stdscr.attron(curses.color_pair(5))
    stdscr.addstr(YCursor, XCursor, "Select an option using the UP/DOWN arrows and ENTER:")
    stdscr.attroff(curses.A_ITALIC)
    stdscr.attroff(curses.color_pair(5))
    XCursor = XCursor + 5    
    YCursor = YCursor + 2

    if optNum == 1:
        stdscr.addstr(YCursor, XCursor, "Enter your Name: " + name, curses.A_STANDOUT)
    else:
        stdscr.addstr(YCursor, XCursor, "Enter your Name: " + name)

    stdscr.refresh()
    k = stdscr.getch()
    if(k != 8):
        name = name + chr(k)

    print(name)
    # Exit the settings
    if k == 27:
        break
    if k == 8:
        if (len(name) > 0):
            name = name[:-1]
    if optNum == 1 and k == 10:
        if name[:-1] != "":
            name = name[:-1]
            path = "Face_Database/" + name+ ".jpg"
            print(path)
            break
        
            