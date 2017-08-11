//
// function/class implete here is relative of session. link to app directly.
//
#define GETTEXT_DOMAIN "rose-lib"

#include "filesystem.hpp"
#include "tensorflow2.hpp"
#include "serialization/string_utils.hpp"
#include "wml_exception.hpp"

#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/io/coded_stream.h>

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

#include <sstream>

namespace tensorflow2 {

class tnull_session_lock
{
public:
	tnull_session_lock(std::unique_ptr<tensorflow::Session>& session)
		: session_(session)
		, ok_(false)
	{}

	~tnull_session_lock()
	{
		if (!ok_) {
			session_.reset(nullptr);
		}
	}
	void set_ok(bool val) { ok_ = val; }

private:
	std::unique_ptr<tensorflow::Session>& session_;
	bool ok_;
};

tensorflow::Status load_model(const std::string& fname, std::unique_ptr<tensorflow::Session>& session) 
{
	tnull_session_lock lock(session);

	tensorflow::SessionOptions options;

	tensorflow::Session* session_pointer = nullptr;
	tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
	if (!session_status.ok()) {
		LOG(ERROR) << "Could not create TensorFlow Session: " << session_status;
		return session_status;
	}
	session.reset(session_pointer);

	bool proto_is_metagraph = false;
	size_t pos = fname.rfind(".pb");
	if (pos == std::string::npos || pos + 3 != fname.size()) {
		pos = fname.rfind(".ckpt.meta");
		VALIDATE(pos != std::string::npos && pos + 10 == fname.size(), null_str);
		proto_is_metagraph = true;
	}

	tensorflow::MetaGraphDef tensorflow_metagraph;
	tensorflow::GraphDef tensorflow_graph2;

	bool read_proto_succeeded = SDL_IsFile(fname.c_str());
	if (read_proto_succeeded) {
		if (proto_is_metagraph) {
			read_proto_succeeded = tensorflow2::read_file_to_proto(fname, tensorflow_metagraph);
		} else {
			read_proto_succeeded = tensorflow2::read_file_to_proto(fname, tensorflow_graph2);
		}
	}
	if (!read_proto_succeeded) {
		LOG(ERROR) << "Failed to load model proto from" << fname;
		return tensorflow::errors::NotFound(fname);
	}
	const tensorflow::GraphDef& tensorflow_graph = proto_is_metagraph? tensorflow_metagraph.graph_def(): tensorflow_graph2;

	tensorflow::Status create_status = session->Create(tensorflow_graph);
	if (!create_status.ok()) {
		LOG(ERROR) << "Could not create TensorFlow Graph: " << create_status;
		return create_status;
	}

	lock.set_ok(true);
	return tensorflow::Status::OK();
}

tensorflow::Status load_model(const std::string& fname, std::pair<std::string, std::unique_ptr<tensorflow::Session> >& session2) 
{
	std::unique_ptr<tensorflow::Session>& session = session2.second;
	const std::string key = file_name(fname);
	if (key != session2.first) {
		tensorflow::Status s = tensorflow2::load_model(fname, session);
		if (!s.ok()) {
			session2.first.clear();
			return s;
		}
		session2.first = key;
	} else {
		VALIDATE(session2.second.get(), null_str);
	}
	return tensorflow::Status::OK();
}

}