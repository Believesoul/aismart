#ifndef LIBROSE_TENSORFLOW2_HPP_INCLUDED
#define LIBROSE_TENSORFLOW2_HPP_INCLUDED

#include <google/protobuf/message_lite.h>
#include <tensorflow/core/public/session.h>

namespace tensorflow2 {
std::string generate_link_function_name(const std::string& dir, const std::string& file);
std::string insert_link_function(const std::string& fullname);
bool read_file_to_proto(const std::string& file_name, ::google::protobuf::MessageLite& proto);

tensorflow::Status load_model(const std::string& fname, std::unique_ptr<tensorflow::Session>& session);
tensorflow::Status load_model(const std::string& fname, std::pair<std::string, std::unique_ptr<tensorflow::Session> >& session2) ;

}

#endif
